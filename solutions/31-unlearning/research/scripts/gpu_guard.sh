#!/usr/bin/env bash
# Garde GPU locale pour une machine perso (une seule carte, partagée avec l'affichage), l'équivalent
# local de `jarvis-heavy` et de la file d'équipe JURECA. Le Makefile la met devant les cibles
# lourdes quand `nvidia-smi` est présent et `jarvis-heavy` absent :
#
#     scripts/gpu_guard.sh <commande> [arguments...]
#
#   1. un job GPU à la fois : verrou flock, les suivants attendent leur tour (jamais deux jobs
#      sur la carte, ce qui ferait déborder la VRAM ou chauffer pour rien) ;
#   2. VRAM : refus de démarrer si la mémoire libre est sous GPU_GUARD_MIN_FREE_MIB ;
#   3. chaleur : le job est suspendu (SIGSTOP) quand la carte atteint GPU_GUARD_PAUSE_C, puis repris
#      (SIGCONT) sous GPU_GUARD_RESUME_C. La carte se protège déjà seule (bridage vers 88 °C) ;
#      ceci garde une marge sur les longues files de runs ;
#   4. télémétrie : température, puissance, VRAM, horloge et raisons de bridage toutes les
#      GPU_GUARD_INTERVAL_S secondes dans runs/gpu/<date>.csv, résumé à la fin ;
#   5. priorité CPU et disque basses (nice 10, ionice best-effort 7) : le bureau reste fluide.
#
# Ctrl-C est transmis au job (le lab marque alors le run `failed`). Le code de sortie est celui du job.
set -uo pipefail

PAUSE_C=${GPU_GUARD_PAUSE_C:-83}
RESUME_C=${GPU_GUARD_RESUME_C:-75}
# Pic mesuré sur RTX 5080 (21/11, config full, batch 32) : ~7,3 Go pour le job seul, gold_soft comme
# trust_region ; 9 Go libres laissent ~1,7 Go de marge à l'affichage et au navigateur.
MIN_FREE_MIB=${GPU_GUARD_MIN_FREE_MIB:-9000}
INTERVAL_S=${GPU_GUARD_INTERVAL_S:-5}
LOCK=${GPU_GUARD_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/tempete-gpu.lock}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR=${GPU_GUARD_LOG_DIR:-$ROOT/runs/gpu}
FIELDS=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm,clocks_event_reasons.active

say() { printf '[gpu-guard] %s\n' "$*" >&2; }

[ $# -ge 1 ] || { say "usage : scripts/gpu_guard.sh <commande> [arguments...]"; exit 2; }
# Déjà sous la garde (une cible make en appelle une autre) : pas de second verrou, qui bloquerait.
[ -z "${GPU_GUARD_ACTIVE:-}" ] || exec "$@"
command -v nvidia-smi >/dev/null || { say "nvidia-smi absent : lancement sans garde."; exec "$@"; }
export GPU_GUARD_ACTIVE=1
# Moins de fragmentation du cache CUDA de torch (même calcul, pics de VRAM plus bas).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

exec 9>"$LOCK"
if ! flock -n 9; then
    say "GPU occupé par un autre job (verrou $LOCK) : attente de sa fin..."
    flock 9
fi

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
if [ "$free_mib" -lt "$MIN_FREE_MIB" ]; then
    say "VRAM libre ${free_mib} Mio < ${MIN_FREE_MIB} Mio demandés. Processus sur la carte :"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    say "fermer ce qui occupe la carte, ou baisser GPU_GUARD_MIN_FREE_MIB en connaissance de cause."
    exit 3
fi

mkdir -p "$LOG_DIR"
log="$LOG_DIR/$(date +%Y%m%d-%H%M%S).csv"
printf 'time,%s,thermal_throttle,hw_slowdown,paused\n' "$FIELDS" >"$log"
printf '# %s\n' "$*" >>"$log"
say "VRAM libre ${free_mib} Mio ; pause à ${PAUSE_C} °C, reprise à ${RESUME_C} °C ; télémétrie : $log"

set -m  # le job a son propre groupe de processus : on le suspend et on lui transmet Ctrl-C en bloc
nice -n 10 ionice -c 2 -n 7 "$@" &
job=$!
set +m
paused=0
pauses=0
signal_job() { kill -"$1" -- "-$job" 2>/dev/null; }
# shellcheck disable=SC2317  # appelée par le trap EXIT
finish() { [ "$paused" = 1 ] && signal_job CONT; }
trap finish EXIT
trap 'signal_job CONT; signal_job INT' INT
trap 'signal_job CONT; signal_job TERM' TERM

started=$SECONDS
while kill -0 "$job" 2>/dev/null; do
    sample=$(nvidia-smi --query-gpu="$FIELDS" --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    temp=${sample%%,*}
    if [[ "$temp" =~ ^[0-9]+$ ]]; then
        if [ "$paused" = 0 ] && [ "$temp" -ge "$PAUSE_C" ]; then
            signal_job STOP && paused=1 && pauses=$((pauses + 1))
            say "$(date +%T) GPU à ${temp} °C >= ${PAUSE_C} : job suspendu jusqu'à ${RESUME_C} °C."
        elif [ "$paused" = 1 ] && [ "$temp" -le "$RESUME_C" ]; then
            signal_job CONT && paused=0
            say "$(date +%T) GPU à ${temp} °C : job repris."
        fi
    fi
    # Raisons de bridage (masque nvidia-smi) : 0x20 / 0x40 thermique logiciel / matériel,
    # 0x8 / 0x80 ralentissement / frein matériel (0x4, plafond de puissance, est normal en charge).
    reasons=${sample##*,}
    [[ "$reasons" =~ ^0x[0-9a-fA-F]+$ ]] || reasons=0
    thermal=$(( (reasons & 0x60) != 0 ))
    hw=$(( (reasons & 0x88) != 0 ))
    printf '%s,%s,%s,%s,%s\n' "$(date +%T)" "$sample" "$thermal" "$hw" "$paused" >>"$log"
    sleep "$INTERVAL_S" &
    wait $! 2>/dev/null
done
wait "$job"
code=$?
trap - EXIT

awk -F, -v secs=$((SECONDS - started)) -v pauses="$pauses" -v code="$code" '
    NR > 2 && $2 ~ /^[0-9]+$/ {
        n++; if ($2 > t) t = $2; if ($3 > p) p = $3; if ($5 > m) m = $5; sp += $3
        thermal += $8; hw += $9
    }
    END {
        printf "[gpu-guard] fin (code %d) en %d min %02d s : max %d °C, max %.0f W (moy. %.0f W), " \
               "VRAM max %d Mio (carte entière) ; pauses thermiques : %d ; échantillons bridés (thermique / matériel) : %d / %d\n",
               code, secs / 60, secs % 60, t, p, (n ? sp / n : 0), m, pauses, thermal, hw
    }' "$log" >&2
exit "$code"
