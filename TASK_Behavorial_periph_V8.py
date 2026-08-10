# -*- coding: utf-8 -*-
"""
analyse comportementale projet FVPS - VERSION 2
============================
Experience comportementale en vision peripherique avec 4 familles de conditions :
  - HMstimuli90 : axe horizontal, orientation 90
  - HMstimuli0  : axe horizontal, orientation 0
  - VMstimuli90 : axe vertical,   orientation 90
  - VMstimuli0  : axe vertical,   orientation 0

4 entrainements filtres non comptabilises, puis 20 runs principaux randomises.
Presentation a 6 Hz, 64 s de stimulation par run principal et 20 s par entrainement.
Les fades d'entree et de sortie durent 2 s chacun.

Le degré du filtre des stimuli est fixe a 32.5 dans cette version.
Le script pointe par defaut vers les dossiers du filtre 32.5. Les quatre
chemins d'images sont configures explicitement dans STIMULI_SUBFOLDERS :
ils peuvent donc etre remplaces par d'autres chemins sous EXP_ROOT.

Modulations visuelles appliquées à toutes les conditions filtrees :
  - à chaque présentation d'image, le contraste suit une enveloppe
    sinusoïdale intra-image entre 80 % et 100 %, puis revient à 80 % ;
  - cette modulation de contraste est multipliée par les fades d'entrée et
    de sortie ;
  - à chaque nouvelle image, la taille est tirée aléatoirement parmi
    90 %, 95 %, 100 %, 105 % ou 110 % de la taille originale ;
  - les images originales font 256 x 256 px, donc les tailles affichées
    possibles sont 230.4, 243.2, 256, 268.8 ou 281.6 px ;
  - la même taille est appliquée aux deux positions affichées simultanément
    afin de garder une présentation gauche/droite ou haut/bas symétrique.

Pattern d'apparition des visages :
  - dans les runs principaux, chaque sequence contient 14 visages a detecter
    pendant les periodes de carre gris ;
  - Dans chaque run, il y'a 7 sequences d'inhibition ( cercle gris),
    soit 1 visage aleatoire par periode d'inhibition, a ne pas detecter ;
  - chaque visage a detecter est precede par un bloc aleatoire d'au moins
    15 objets ;
  - les visages a detecter ne peuvent pas apparaitre visuellement dans les
    1,5 s avant ni dans les 1,5 s apres une periode d'inhibition centrale ;
  - les 14 visages detectables en carre gris sont tires uniquement dans les
    intervalles autorises entre ces zones tampon ;
  - les visages d'inhibition restent tires pendant les periodes de cercle gris
    et sont enregistres dans les donnees comme essais gray_circle ( analyse separée );
  - a chaque apparition, la meme image est affichee simultanement aux deux
    positions de la condition (gauche/droite pour HM, haut/bas pour VM).

Tache principale :
  - quand la fixation centrale est un carre gris, le participant appuie sur
    ESPACE s'il pense detecter un visage en peripherie ;
  - un appui sur ESPACE est classe comme hit s'il survient apres un visage
    dans une fenetre de reponse de 1,5 s ;
  - les appuis non associes a un visage sont classes comme fausses alarmes.

Tache d'attention centrale :
  - pendant certaines periodes, le carre gris devient un cercle gris ;
  - quand le cercle gris est affiche, le participant doit inhiber toute
    reponse sur ESPACE, meme si des images apparaissent en peripherie ;
  - il y a 7 periodes d'inhibition de 2 s par run principal et 3 periodes
    d'inhibition de 2 s par entrainement ;
  - si le participant appuie au moins 1 fois sur ESPACE pendant les cercles
    gris d'une sequence, un rappel apparait avant la sequence suivante.

Les donnees sont exportees directement dans deux fichiers Excel :
  - un fichier participant avec les essais gray_square et gray_circle ;
  - un fichier frames/missed frames.
"""

# %%
# ==============================
# Import modules
# ==============================
from psychopy import core, visual, gui, event, monitors, logging
from psychopy.hardware import keyboard
import random
import os
import math
import zipfile
import bisect
import numpy as np
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape

rng = random.SystemRandom()

# %%
# ==============================
# Paramètres facilement modifiables
# ==============================
N_RUNS           = 20        # nombre total de runs principaux
N_PRACTICE_RUNS  = 4         # entrainements filtres non comptabilises
FREQ_HZ          = 6         # fréquence de présentation des stimuli (Hz)
REFRESH_RATE     = 60        # fréquence de rafraîchissement de l'écran (Hz)
MAIN_STIM_DURATION     = 64.0  # duree de stimulation principale, fades inclus
PRACTICE_STIM_DURATION = 20.0  # duree des entrainements, fades inclus
PAUSE_DURATION   = 2.0       # durée de la pause avant chaque séquence (secondes)
FADE_DURATION    = 2.0       # durée du fade-in et du fade-out (secondes)
MAIN_DURATION    = MAIN_STIM_DURATION - (2 * FADE_DURATION)
PRACTICE_MAIN_DURATION = PRACTICE_STIM_DURATION - (2 * FADE_DURATION)
STIM_EVENTS_PER_RUN = int(round(MAIN_STIM_DURATION * FREQ_HZ))
PRACTICE_STIM_EVENTS_PER_RUN = int(round(PRACTICE_STIM_DURATION * FREQ_HZ))

ECC_PIX          = 350.0631883  # excentricité : centre fixation → centre stimulus (px)
STIM_PIX         = 256       # taille des stimuli (px)
STIM_SIZE_FACTORS = [0.90, 0.95, 1.00, 1.05, 1.10]
FIX_PIX          = 12        # taille du point de fixation (px)
FIX_CIRCLE_PIX   = 13.5      # diametre du cercle d'inhibition central (px)

SCREEN_RES       = (1920, 1080)
SCREEN_WIDTH_CM  = 53.2
VIEW_DIST_CM     = 100.0
BG_COLOR         = '#737373'
FIX_COLOR        = '#A8A8A8'  # type de gris du carre/cercle
FIX_COLOR2       = '#D8D8D8' # couleur du texte detection
FIXATION_SQUARE_LABEL = 'gray_square'
FIXATION_CIRCLE_LABEL = 'gray_circle'
SUCCESS_COLOR    = 'green'

# Seuil de détection des missed frames (en secondes)
# Une frame a 60 Hz dure ~16.67 ms ; on signale si > 1.2x la duree theorique
FRAME_DURATION_EXPECTED = 1.0 / REFRESH_RATE
MISSED_FRAME_THRESHOLD  = FRAME_DURATION_EXPECTED * 1.2

# Types de stimuli melanges selon les fichiers reels des dossiers.
STIM_CATEGORIES = ['faces', 'objects']
FACE_RUN_COUNTS = [14]
FACE_COUNT_BLOCK_REPETITIONS = 20
INHIBITION_FACE_RUN_COUNT = 7
PRACTICE_FACE_RUN_COUNTS = [4]
MIN_OBJECTS_BEFORE_FACE = 15
RESPONSE_WINDOW_SEC = 1.5
FADE_IN_MISS_EXEMPTION_SEC = 0.5
FADE_OUT_MISS_EXEMPTION_START_SEC = 1.5
FADE_MISS_EXEMPT_CLASSIFICATION = 'not_scored_fade'
STIM_INTERVAL_EXPECTED = 1.0 / FREQ_HZ
STIM_INTERVAL_TOLERANCE = FRAME_DURATION_EXPECTED * 0.5

# Tache d'attention centrale : inhibition de la reponse ESPACE quand
# le carre de fixation devient un cercle gris.
ATTENTION_INHIBITION_EVENTS = 7
PRACTICE_ATTENTION_INHIBITION_EVENTS = 3
ATTENTION_INHIBITION_DURATION = 2.0
ATTENTION_MIN_STIM_OFFSET = 3.0
ATTENTION_WARNING_FALSE_ALARMS = 1
ATTENTION_MIN_GO_BETWEEN_INHIBITIONS = 4.0
ATTENTION_MIN_ONSET_SPACING = ATTENTION_INHIBITION_DURATION + ATTENTION_MIN_GO_BETWEEN_INHIBITIONS

# Contraste pilote par une enveloppe sinusoidale intra-image, multipliee
# par l'enveloppe lineaire de fade-in/fade-out.
STIM_CONTRAST_MIN = 0.80
STIM_CONTRAST_MAX = 1.00

# Délai double-Échap (secondes)
ESC_DOUBLE_DELAY = 2.0

# Racine commune des stimuli.
EXP_ROOT = "C:/Users/milan/OneDrive - UCL/d-memoire/exp_comportementale"
FILTER_DEGREE = '32.5'
FILTER_OUTPUT_DIR = {'data': 'datafilter32.5', 'frames': 'frames32.5'}
STIMULI_SUBFOLDERS = {
    'ori0_faces': 'stimuli_filter_32.5/ori0_faces_32.5',
    'ori0_objects': 'stimuli_filter_32.5/ori0_objects_32.5',
    'ori90_faces': 'stimuli_filter_32.5/ori90_faces_32.5',
    'ori90_objects': 'stimuli_filter_32.5/ori90_objects_32.5',
}

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')

# %%
# ==============================
# Calcul géométrique (informatif)
# ==============================
px_per_cm = SCREEN_RES[0] / SCREEN_WIDTH_CM
ecc_cm    = ECC_PIX / px_per_cm
ecc_deg   = math.degrees(math.atan(ecc_cm / VIEW_DIST_CM))
stim_cm   = STIM_PIX / px_per_cm
stim_deg  = math.degrees(2 * math.atan((stim_cm / 2) / VIEW_DIST_CM))

print("=" * 54)
print("CALIBRATION GÉOMÉTRIQUE")
print("=" * 54)
print(f"Résolution écran      : {SCREEN_RES[0]} x {SCREEN_RES[1]} px")
print(f"Largeur écran         : {SCREEN_WIDTH_CM:.2f} cm")
print(f"Distance observation  : {VIEW_DIST_CM:.2f} cm")
print(f"Pixels par cm         : {px_per_cm:.3f} px/cm")
print(f"Excentricité          : {ECC_PIX} px = {ecc_cm:.3f} cm = {ecc_deg:.3f}°")
print(f"Taille stimulus       : {STIM_PIX} px = {stim_deg:.3f}°")
print("=" * 54)

# %%
# ==============================
# Boîte de dialogue
# ==============================
exp_name = 'analyse comportementale projet FVPS'
exp_info = {
    'participant':     '',
    'Entrainements': ['Yes', 'No'],
    'Preview_stimuli': ['Yes', 'No'],
}

dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name)
if not dlg.OK:
    core.quit()

DO_PRACTICE = (str(exp_info['Entrainements']).strip() == 'Yes')
participant = str(exp_info['participant']).strip() or 'participant'
session_dt  = datetime.now().strftime('%Y%m%d_%H%M%S')

# %%
# ==============================
# Chemins de sortie
# ==============================
output_dirs = FILTER_OUTPUT_DIR
DATA_DIR = f"{EXP_ROOT}/analyses stats/{output_dirs['data']}"
MAIN_DATA_DIR = f"{DATA_DIR}/output comportemental"
FRAME_DATA_DIR = f"{DATA_DIR}/{output_dirs['frames']}"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MAIN_DATA_DIR, exist_ok=True)
os.makedirs(FRAME_DATA_DIR, exist_ok=True)

out_xlsx_fname = f"{MAIN_DATA_DIR}/{participant}_{session_dt}.xlsx"
log_xlsx_fname = f"{FRAME_DATA_DIR}/{participant}_{session_dt}_frames.xlsx"

# %%
# ==============================
# Chemins stimuli
# ==============================
def resolve_stim_folder(folder_key):
    folder_name = STIMULI_SUBFOLDERS[folder_key]
    folder = folder_name if os.path.isabs(folder_name) else os.path.join(EXP_ROOT, folder_name)
    if os.path.isdir(folder):
        return folder
    raise FileNotFoundError(
        "Aucun dossier stimuli trouve pour "
        f"{folder_key}. Dossier configure : {folder}")

FOLDER_ORI0_FACES   = resolve_stim_folder('ori0_faces')
FOLDER_ORI0_OBJECTS = resolve_stim_folder('ori0_objects')
FOLDER_ORI90_FACES  = resolve_stim_folder('ori90_faces')
FOLDER_ORI90_OBJECTS= resolve_stim_folder('ori90_objects')

print(f"Filtre stimuli choisi : {FILTER_DEGREE}")
print(f"Racine stimuli        : {EXP_ROOT}")
print("Dossiers stimuli configures :")
for folder_key, folder_name in STIMULI_SUBFOLDERS.items():
    folder = folder_name if os.path.isabs(folder_name) else os.path.join(EXP_ROOT, folder_name)
    print(f"  {folder_key}: {folder}")

# %%
# ==============================
# Fonctions utilitaires
# ==============================
def list_images(folder):
    if not os.path.isdir(folder):
        print(f"DOSSIER INTROUVABLE : {folder}")
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(IMG_EXT)]

def safe_filename(path):
    return os.path.basename(path).replace(',', '.')

# %%
# ==============================
# Chargement des stimuli
# ==============================
pools = {
    0: {'faces': list_images(FOLDER_ORI0_FACES), 'objects': list_images(FOLDER_ORI0_OBJECTS)},
    90: {'faces': list_images(FOLDER_ORI90_FACES), 'objects': list_images(FOLDER_ORI90_OBJECTS)}}

for ori in [0, 90]:
    for cat in ['faces', 'objects']:
        if len(pools[ori][cat]) == 0:
            raise FileNotFoundError(
                f"Aucune image trouvée pour ori={ori}, catégorie={cat}.\n"
                "Vérifiez les chemins configures dans STIMULI_SUBFOLDERS.")

print("STIMULI CHARGÉS")
print(f"  ori0_faces    : {len(pools[0]['faces'])} images")
print(f"  ori0_objects  : {len(pools[0]['objects'])} images")
print(f"  ori90_faces   : {len(pools[90]['faces'])} images")
print(f"  ori90_objects : {len(pools[90]['objects'])} images")

# %%
# ==============================
# Definition technique des conditions et variantes de presentation
# ==============================
CONDITIONS = {
    'HMstimuli90': {'axis': 'horizontal', 'ori': 90, 'pos1_name': 'left', 'pos2_name': 'right', 'pos1': (-ECC_PIX, 0), 'pos2': (ECC_PIX, 0), 'trigger_code': 11},
    'HMstimuli0':  {'axis': 'horizontal', 'ori': 0,  'pos1_name': 'left', 'pos2_name': 'right', 'pos1': (-ECC_PIX, 0), 'pos2': (ECC_PIX, 0), 'trigger_code': 12},
    'VMstimuli90': {'axis': 'vertical',   'ori': 90, 'pos1_name': 'top',  'pos2_name': 'bottom', 'pos1': (0, ECC_PIX),  'pos2': (0, -ECC_PIX), 'trigger_code': 21},
    'VMstimuli0':  {'axis': 'vertical',   'ori': 0,  'pos1_name': 'top',  'pos2_name': 'bottom', 'pos1': (0, ECC_PIX),  'pos2': (0, -ECC_PIX), 'trigger_code': 22},
}

# %%
# ==============================
# Sequence randomisee des 20 runs principaux
# ==============================
run_sequence = (
    ['HMstimuli90'] * 5 +
    ['HMstimuli0'] * 5 +
    ['VMstimuli90'] * 5 +
    ['VMstimuli0'] * 5
)
assert N_RUNS == len(run_sequence), "N_RUNS doit correspondre au plan remanie."
rng.shuffle(run_sequence)

main_detectable_face_count_plan = []
assert N_RUNS == len(FACE_RUN_COUNTS) * FACE_COUNT_BLOCK_REPETITIONS, (
    "N_RUNS doit correspondre a 20 runs de 14 visages detectables.")
for block_i in range(FACE_COUNT_BLOCK_REPETITIONS):
    face_count_block = FACE_RUN_COUNTS[:]
    rng.shuffle(face_count_block)
    main_detectable_face_count_plan.extend(face_count_block)
print(f"\nOrdre des {N_RUNS} runs (randomisé) :")
for i, c in enumerate(run_sequence, 1):
    print(
        f"  Run {i:02d} : {c} | "
        f"{main_detectable_face_count_plan[i - 1]} visages detectables + "
        f"{INHIBITION_FACE_RUN_COUNT} visages inhibition")


# %%
# ==============================
# Timing
# ==============================
FrameN         = int(round(REFRESH_RATE / FREQ_HZ))   # frames par stimuli
TOTAL_DURATION = PAUSE_DURATION + FADE_DURATION + MAIN_DURATION + FADE_DURATION
TOTAL_FRAMES   = int(round(TOTAL_DURATION * REFRESH_RATE))

if FrameN > 1:
    STIM_CONTRAST_SINE_PEAK = max(
        math.sin(math.pi * i / (FrameN - 1)) for i in range(FrameN))
else:
    STIM_CONTRAST_SINE_PEAK = 1.0

def sinusoidal_stim_contrast(frame_n):
    """Contraste 80-100-80 %, recalcule a chaque presentation d'image."""
    if FrameN <= 1:
        return STIM_CONTRAST_MAX
    stim_frame = frame_n % FrameN
    phase = stim_frame / (FrameN - 1)
    modulation = math.sin(math.pi * phase) / STIM_CONTRAST_SINE_PEAK
    return STIM_CONTRAST_MIN + (
        (STIM_CONTRAST_MAX - STIM_CONTRAST_MIN) * modulation)

def draw_stim_size():
    """Tire une taille 90-110 % par increments de 5 %."""
    scale = rng.choice(STIM_SIZE_FACTORS)
    size = STIM_PIX * scale
    return (size, size), scale

# %%
# ==============================
# Logique de randomisation des images
# ==============================
def planned_stim_onset(index):
    return PAUSE_DURATION + (index / FREQ_HZ)

def face_allowed_for_detection(index, attention_events, stim_duration):
    onset = planned_stim_onset(index)
    offset = onset + STIM_INTERVAL_EXPECTED
    for ev in attention_events or []:
        buffer_start = ev['onset'] - RESPONSE_WINDOW_SEC
        buffer_end = ev['offset'] + RESPONSE_WINDOW_SEC
        if offset > buffer_start and onset < buffer_end:
            return False
    time_in_stimulation = index / FREQ_HZ
    if is_fade_miss_exempt(time_in_stimulation, stim_duration):
        return False
    return True

def select_detectable_face_indices(total_events, face_count, attention_events, stim_duration):
    allowed_indices = sorted(
        idx for idx in range(MIN_OBJECTS_BEFORE_FACE, total_events)
        if face_allowed_for_detection(idx, attention_events, stim_duration)
    )
    if face_count == 0:
        return []
    if len(allowed_indices) < face_count:
        return None

    gap = MIN_OBJECTS_BEFORE_FACE + 1
    n_allowed = len(allowed_indices)
    next_positions = [
        bisect.bisect_left(allowed_indices, idx + gap)
        for idx in allowed_indices
    ]

    can_place = [[False] * (face_count + 1) for _ in range(n_allowed + 1)]
    for i in range(n_allowed + 1):
        can_place[i][0] = True
    for i in range(n_allowed - 1, -1, -1):
        for remaining_faces in range(1, face_count + 1):
            can_place[i][remaining_faces] = (
                can_place[i + 1][remaining_faces]
                or can_place[next_positions[i]][remaining_faces - 1]
            )

    if not can_place[0][face_count]:
        return None

    selected = []
    current_pos = 0
    remaining_faces = face_count
    while remaining_faces > 0:
        candidate_positions = [
            i for i in range(current_pos, n_allowed)
            if can_place[next_positions[i]][remaining_faces - 1]
        ]
        if not candidate_positions:
            return None
        chosen_pos = rng.choice(candidate_positions)
        selected.append(allowed_indices[chosen_pos])
        current_pos = next_positions[chosen_pos]
        remaining_faces -= 1

    return selected

def inhibition_indices_for_event(total_events, attention_event):
    return [
        idx for idx in range(total_events)
        if (
            attention_event['onset'] + STIM_INTERVAL_EXPECTED
            <= planned_stim_onset(idx)
            < attention_event['offset'] - STIM_INTERVAL_EXPECTED
        )
    ]

def select_inhibition_face_indices(total_events, face_count, attention_events):
    if face_count == 0:
        return []
    attention_events = attention_events or []
    per_event_indices = [
        inhibition_indices_for_event(total_events, ev)
        for ev in attention_events
    ]
    per_event_indices = [indices for indices in per_event_indices if indices]
    all_indices = sorted({idx for indices in per_event_indices for idx in indices})
    if len(all_indices) < face_count:
        return None

    selected = set()
    remaining = face_count

    if face_count >= len(per_event_indices):
        event_order = list(range(len(per_event_indices)))
        rng.shuffle(event_order)
        for event_i in event_order:
            available = [idx for idx in per_event_indices[event_i] if idx not in selected]
            if not available:
                continue
            selected.add(rng.choice(available))
            remaining -= 1

    available_indices = [idx for idx in all_indices if idx not in selected]
    rng.shuffle(available_indices)
    for idx in available_indices:
        if remaining == 0:
            break
        selected.add(idx)
        remaining -= 1

    if remaining != 0:
        return None
    return sorted(selected)

def build_category_sequence(
        total_events, detectable_face_count, attention_events=None,
        inhibition_face_count=0, stim_duration=MAIN_STIM_DURATION):
    """Cree une sequence objets/visages avec cibles et inhibitions separees."""
    attention_events = attention_events or []
    for _ in range(200):
        detectable_face_indices = select_detectable_face_indices(
            total_events, detectable_face_count, attention_events, stim_duration)
        if detectable_face_indices is None:
            continue
        inhibition_face_indices = select_inhibition_face_indices(
            total_events, inhibition_face_count, attention_events)
        if inhibition_face_indices is None:
            continue

        overlapping_indices = set(detectable_face_indices) & set(inhibition_face_indices)
        if overlapping_indices:
            continue

        sequence = ['objects'] * total_events
        for idx in detectable_face_indices:
            sequence[idx] = 'faces'
        for idx in inhibition_face_indices:
            sequence[idx] = 'faces'
        return sequence, detectable_face_count, inhibition_face_count

    raise RuntimeError(
        "Impossible de generer une sequence avec les nombres fixes de visages "
        "detectables et de visages d'inhibition.")

def get_image_pool(ori, category):
    return pools[ori][category]

def build_image_deck(ori, category):
    """Melange les fichiers d'une categorie et d'une orientation."""
    deck = get_image_pool(ori, category)[:]
    rng.shuffle(deck)
    return deck

def draw_image(image_decks, ori, category):
    """Tire une image sans repetition avant epuisement du paquet."""
    if not image_decks[category]:
        image_decks[category].extend(build_image_deck(ori, category))
    return image_decks[category].pop()

# %%
# ==============================
# Fenêtre PsychoPy
# ==============================
mon = monitors.Monitor('mon1')
mon.setSizePix(SCREEN_RES)
mon.setWidth(SCREEN_WIDTH_CM)
mon.setDistance(VIEW_DIST_CM)

def choose_experiment_screen():
    """Utilise l'ecran externe si PsychoPy/pyglet en detecte plusieurs."""
    try:
        import pyglet
        if hasattr(pyglet, 'canvas'):
            screens = pyglet.canvas.get_display().get_screens()
        else:
            from pyglet import display
            screens = display.get_display().get_screens()
        return max(0, len(screens) - 1)
    except Exception as exc:
        print(f"Detection ecrans impossible, utilisation ecran 0 : {exc}")
        return 0

EXPERIMENT_SCREEN = choose_experiment_screen()
print(f"Ecran PsychoPy utilise : {EXPERIMENT_SCREEN}")
win = visual.Window(
    monitor=mon, size=SCREEN_RES, color=BG_COLOR, units='pix',
    fullscr=True, screen=EXPERIMENT_SCREEN, allowGUI=False)
win.mouseVisible = False
logging.console.setLevel(logging.WARNING)

kb    = keyboard.Keyboard()
timer = core.Clock()

# %%
# ==============================
# Point de fixation et signal central d'inhibition
# ==============================
fixation = visual.Rect(win=win, units='pix', width=FIX_PIX, height=FIX_PIX, fillColor=FIX_COLOR, lineColor=FIX_COLOR, pos=(0, 0))
fixation_inhibition = visual.Circle(win=win, units='pix', radius=FIX_CIRCLE_PIX / 2.0, edges=64, fillColor=FIX_COLOR, lineColor=FIX_COLOR, pos=(0, 0))

# %%
# ==============================
# Cache des images
# ==============================
image_cache = {}
for ori in [0, 90]:
    for cat in ['faces', 'objects']:
        for p in pools[ori][cat]:
            if p not in image_cache:
                image_cache[p] = visual.ImageStim(win=win, image=p, units='pix', size=(STIM_PIX, STIM_PIX))

# %%
# ==============================
# Fichiers de sortie – en-têtes
# ==============================
MAIN_HEADER = [
    'ID_personne_numero_session', 'condition_name',
    'stimulus_filter_status', 'filter_degree',
    'presentation_axis', 'stimulus_orientation',
    'fixation_condition',
    'stimulus_category_face_or_object',
    'stimulus_onset_time_sec_from_run_start',
    'participant_response_key', 'reaction_time_sec_from_this_onset',
    'behavioral_classification',
]
main_rows = []

FRAME_HEADER = [
    'participant_id', 'condition_name',
    'stimulus_filter_status', 'filter_degree',
    'presentation_axis', 'stimulus_orientation',
    'measured_frame_duration_ms_mean', 'measured_frame_duration_ms_std',
    'expected_frame_duration_ms', 'six_hz_not_ok_count',
]
frame_rows = []

def condition_excel_fields(condition_name):
    cfg = CONDITIONS[condition_name]
    return ['filtered', FILTER_DEGREE, cfg['axis'], cfg['ori']]

def append_main_row(row):
    main_rows.append(row)

def append_frame_row(row):
    frame_rows.append(row)

def append_frame_summary_row(condition_name, frame_duration_ms_values, timing_summary):
    if frame_duration_ms_values:
        mean_ms = sum(frame_duration_ms_values) / len(frame_duration_ms_values)
        variance_ms = sum((value - mean_ms) ** 2 for value in frame_duration_ms_values) / len(frame_duration_ms_values)
        std_ms = math.sqrt(variance_ms)
        mean_ms = round(mean_ms, 3)
        std_ms = round(std_ms, 3)
    else:
        mean_ms = ''
        std_ms = ''

    missing_or_extra_stimuli = abs(timing_summary['stim_count'] - STIM_EVENTS_PER_RUN)
    six_hz_not_ok_count = timing_summary['bad'] + missing_or_extra_stimuli
    append_frame_row([
        participant, logged_condition_name(condition_name),
        *condition_excel_fields(condition_name), mean_ms, std_ms,
        round(FRAME_DURATION_EXPECTED * 1000.0, 3), six_hz_not_ok_count])

def _xlsx_col_name(col_index):
    name = ''
    while col_index:
        col_index, remainder = divmod(col_index - 1, 26)
        name = chr(65 + remainder) + name
    return name

def _xml_attr(text):
    return xml_escape(str(text), {'"': '&quot;'})

def _xlsx_cell(cell_ref, value, style_id):
    if value is None or value == '':
        return f'<c r="{cell_ref}" s="{style_id}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}" s="{style_id}"><v>{value}</v></c>'
    value_text = xml_escape(str(value))
    return (
        f'<c r="{cell_ref}" t="inlineStr" s="{style_id}">'
        f'<is><t>{value_text}</t></is></c>')

def _xlsx_row(row_index, values, style_id):
    cells = []
    for col_index, value in enumerate(values, start=1):
        cell_ref = f'{_xlsx_col_name(col_index)}{row_index}'
        cells.append(_xlsx_cell(cell_ref, value, style_id))
    return f'<row r="{row_index}">{"".join(cells)}</row>'

def write_xlsx_table(fpath, headers, rows, sheet_name='Frames'):
    """Ecrit un .xlsx simple avec colonnes alignees et en-tete fige."""
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    n_cols = len(headers)
    n_rows = len(rows) + 1
    last_col = _xlsx_col_name(n_cols)

    widths = []
    for col_index, header in enumerate(headers):
        max_len = len(str(header))
        for row in rows:
            if col_index < len(row):
                max_len = max(max_len, len(str(row[col_index])))
        widths.append(min(max(max_len + 2, 10), 45))

    content_types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>'
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    workbook = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{_xml_attr(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    workbook_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left style="thin"><color rgb="FFB7B7B7"/></left><right style="thin"><color rgb="FFB7B7B7"/></right><top style="thin"><color rgb="FFB7B7B7"/></top><bottom style="thin"><color rgb="FFB7B7B7"/></bottom><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'

    with zipfile.ZipFile(fpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types)
        zf.writestr('_rels/.rels', rels)
        zf.writestr('xl/workbook.xml', workbook)
        zf.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
        zf.writestr('xl/styles.xml', styles)
        with zf.open('xl/worksheets/sheet1.xml', 'w') as sheet:
            def write(text):
                sheet.write(text.encode('utf-8'))

            write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
            write('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ')
            write('xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">')
            write(f'<dimension ref="A1:{last_col}{n_rows}"/>')
            write('<sheetViews><sheetView workbookViewId="0">')
            write('<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>')
            write('<selection pane="bottomLeft"/></sheetView></sheetViews>')
            write('<cols>')
            for col_index, width in enumerate(widths, start=1):
                write(f'<col min="{col_index}" max="{col_index}" width="{width}" customWidth="1"/>')
            write('</cols><sheetData>')
            write(_xlsx_row(1, headers, 1))
            for row_index, row in enumerate(rows, start=2):
                write(_xlsx_row(row_index, row, 2))
            write('</sheetData>')
            write(f'<autoFilter ref="A1:{last_col}{n_rows}"/>')
            write('</worksheet>')

def save_frame_workbook():
    try:
        write_xlsx_table(log_xlsx_fname, FRAME_HEADER, frame_rows, sheet_name='Frames')
        print(f"XLSX frames     : {log_xlsx_fname}")
        return True
    except Exception as exc:
        print(f"Export XLSX frames impossible : {exc}")
        return False

def save_main_workbook():
    try:
        write_xlsx_table(out_xlsx_fname, MAIN_HEADER, main_rows, sheet_name='Participant_clean')
        print(f"XLSX principal  : {out_xlsx_fname}")
        return True
    except Exception as exc:
        print(f"Export XLSX principal impossible : {exc}")
        return False

def save_workbooks():
    main_saved = save_main_workbook()
    frames_saved = save_frame_workbook()
    return main_saved, frames_saved

print(f"\nXLSX principal  : {out_xlsx_fname}")
print(f"XLSX frames     : {log_xlsx_fname}")

# %%
# ==============================
# Gestion de l'échappement (double-Échap)
# ==============================
esc_state = {'first_press_time': None, 'warning_shown': False}

esc_warning_stim = visual.TextStim(
    win=win,
    text="Appuyez encore une fois sur Échap pour quitter",
    color='yellow',
    height=30,
    units='pix',
    pos=(0, -200))

def check_escape(draw_warning=True):
    """
    Retourne True si l'expérience doit s'arrêter.
    Gère la logique double-Échap avec délai de 2 secondes.
    """
    keys = event.getKeys(keyList=['escape'])
    now  = core.getTime()

    if 'escape' in keys:
        if esc_state['first_press_time'] is None:
            # Premier appui
            esc_state['first_press_time'] = now
            esc_state['warning_shown']    = False
        else:
            # Deuxième appui – vérifier délai
            if now - esc_state['first_press_time'] <= ESC_DOUBLE_DELAY:
                return True  # → quitter
            else:
                # Trop tard : réinitialiser et compter comme premier appui
                esc_state['first_press_time'] = now
                esc_state['warning_shown']    = False
    else:
        # Réinitialiser si le délai est expiré
        if (esc_state['first_press_time'] is not None and
                now - esc_state['first_press_time'] > ESC_DOUBLE_DELAY):
            esc_state['first_press_time'] = None
            esc_state['warning_shown']    = False

    return False

def esc_warning_active():
    """Vrai si le premier Échap est actif et non expiré."""
    if esc_state['first_press_time'] is None:
        return False
    return (core.getTime() - esc_state['first_press_time']) <= ESC_DOUBLE_DELAY

def clean_quit():
    """Ferme proprement après sauvegarde des données."""
    save_workbooks()
    win.close()
    core.quit()

# %%
# ==============================
# Fonctions d'affichage
# ==============================
def draw_fixation():
    fixation.draw()

def active_attention_event(t_seq, attention_events):
    for ev in attention_events:
        if ev['onset'] <= t_seq < ev['offset']:
            return ev
    return None

def is_in_attention_window(t_abs, attention_events):
    return active_attention_event(t_abs, attention_events) is not None

def draw_fixation_marker(t_seq, attention_events):
    ev = active_attention_event(t_seq, attention_events)
    if ev is None:
        fixation.draw()
    else:
        fixation_inhibition.draw()

def draw_centered_text(text, color='white', height=26, pos=(0, 0), bold=False, wrap=0.85):
    visual.TextStim(win=win, text=text, color=color, height=height, bold=bold,
                    wrapWidth=SCREEN_RES[0] * wrap, units='pix', pos=pos).draw()

def show_text(text, wait=True, with_fixation=False):
    draw_centered_text(text)
    if with_fixation:
        draw_fixation()
    win.flip()
    if wait:
        keys = event.waitKeys(keyList=['space', 'escape'])
        if keys and 'escape' in keys:
            clean_quit()

def show_run_start(text, warn_fixation=False, highlight=None, warning_text=None):
    main_y = -75 if highlight else -35
    highlight_y = 160 if warn_fixation else 245
    warning_y = 245 if highlight else 170
    draw_centered_text(text, pos=(0, main_y))
    if highlight:
        draw_centered_text(highlight['text'], color=highlight['color'], height=32,
                           pos=(0, highlight_y), bold=True, wrap=0.9)
    if warn_fixation:
        draw_centered_text(warning_text or "ATTENTION : VOUS NE DEVIEZ PAS APPUYER",
                           color=FIX_COLOR2, height=36, pos=(0, warning_y), bold=True, wrap=0.9)
    win.flip()
    keys = event.waitKeys(keyList=['space', 'escape'])
    if keys and 'escape' in keys:
        clean_quit()

def logged_condition_name(condition_name):
    return condition_name

def run_start_text(run_label, condition_name, stim_duration):
    return f"{run_label}\n\nCondition : {condition_name}\nDuree : {stim_duration:.0f} s\n\nESPACE = commencer"

def format_face_detection_percent(response_summary):
    faces_total = response_summary.get('planned_faces', response_summary['faces'])
    percent = 0.0 if faces_total == 0 else (response_summary['hits'] / faces_total) * 100.0
    return {'text': f"DETECTION CORRECTE DE VISAGES : {percent:.0f}%", 'color': SUCCESS_COLOR if percent > 75.0 else FIX_COLOR2}

def format_attention_warning(false_alarm_count):
    return f"ATTENTION : VOUS NE DEVIEZ PAS APPUYER {false_alarm_count} FOIS"

def summarize_stim_timing(stim_onsets):
    intervals = [later - earlier for earlier, later in zip(stim_onsets[:-1], stim_onsets[1:])]
    if not intervals:
        return {'ok': False, 'stim_count': len(stim_onsets), 'total': 0, 'bad': 0,
                'mean_hz': 0.0, 'max_dev_ms': 0.0,
                'expected_ms': STIM_INTERVAL_EXPECTED * 1000.0, 'tolerance_ms': STIM_INTERVAL_TOLERANCE * 1000.0}

    deviations = [abs(interval - STIM_INTERVAL_EXPECTED) for interval in intervals]
    bad_count = sum(dev > STIM_INTERVAL_TOLERANCE for dev in deviations)
    mean_interval = sum(intervals) / len(intervals)
    return {'ok': bad_count == 0, 'stim_count': len(stim_onsets), 'total': len(intervals), 'bad': bad_count,
            'mean_hz': 1.0 / mean_interval if mean_interval > 0 else 0.0,
            'max_dev_ms': max(deviations) * 1000.0, 'expected_ms': STIM_INTERVAL_EXPECTED * 1000.0,
            'tolerance_ms': STIM_INTERVAL_TOLERANCE * 1000.0}

def is_fade_miss_exempt(time_in_stimulation, stim_duration):
    """True pour les visages non detectes a ne pas compter comme miss."""
    fade_out_start = stim_duration - FADE_DURATION
    return (
        time_in_stimulation <= FADE_IN_MISS_EXEMPTION_SEC
        or time_in_stimulation >= fade_out_start + FADE_OUT_MISS_EXEMPTION_START_SEC
    )

def classify_events(stim_events, response_times, attention_events=None):
    sorted_onsets = sorted(stim_events)
    used_response_ids = set()
    face_matched_response_ids = set()
    attention_events = attention_events or []

    for idx, rt_abs in enumerate(response_times):
        if not is_in_attention_window(rt_abs, attention_events):
            continue
        inhibition_candidates = [
            ons for ons in sorted_onsets
            if ons <= rt_abs and rt_abs - ons <= RESPONSE_WINDOW_SEC
            and stim_events[ons].get('fixation_condition') == FIXATION_CIRCLE_LABEL
            and not stim_events[ons]['responded']
        ]
        if inhibition_candidates:
            ons = inhibition_candidates[-1]
            ev = stim_events[ons]
            ev['responded'] = True
            ev['rt'] = round(rt_abs - ons, 4)
            ev['classification'] = 'false_alarm'
        used_response_ids.add(idx)

    for ons in sorted_onsets:
        ev = stim_events[ons]
        if not ev['face_present'] or ev.get('fixation_condition') != FIXATION_SQUARE_LABEL:
            continue
        valid_responses = [(idx, rt_abs) for idx, rt_abs in enumerate(response_times)
                           if idx not in used_response_ids
                           and rt_abs > ons and rt_abs - ons <= RESPONSE_WINDOW_SEC
                           and not is_in_attention_window(rt_abs, attention_events)]
        if valid_responses:
            idx, rt_abs = valid_responses[0]
            used_response_ids.add(idx)
            face_matched_response_ids.add(idx)
            ev['responded'] = True
            ev['rt'] = round(rt_abs - ons, 4)
            ev['classification'] = 'hit'
        else:
            if ev.get('miss_exempt_due_to_fade', False):
                ev['classification'] = FADE_MISS_EXEMPT_CLASSIFICATION
            else:
                ev['classification'] = 'miss'

    for idx, rt_abs in enumerate(response_times):
        if idx in used_response_ids:
            continue
        if is_in_attention_window(rt_abs, attention_events):
            continue
        object_candidates = [ons for ons in sorted_onsets
                             if ons <= rt_abs and rt_abs - ons <= RESPONSE_WINDOW_SEC
                             and stim_events[ons].get('fixation_condition') == FIXATION_SQUARE_LABEL
                             and not stim_events[ons]['face_present'] and not stim_events[ons]['responded']]
        if object_candidates:
            ons = object_candidates[-1]
            ev = stim_events[ons]
            ev['responded'] = True
            ev['rt'] = round(rt_abs - ons, 4)
            ev['classification'] = 'false_alarm'
            used_response_ids.add(idx)

    for ons in sorted_onsets:
        ev = stim_events[ons]
        if ev['classification'] is None:
            if ev.get('fixation_condition') == FIXATION_CIRCLE_LABEL:
                ev['classification'] = 'correct_inhibition'
            elif ev['face_present']:
                if ev.get('miss_exempt_due_to_fade', False):
                    ev['classification'] = FADE_MISS_EXEMPT_CLASSIFICATION
                else:
                    ev['classification'] = 'miss'
            else:
                ev['classification'] = 'correct_rejection'

    return stim_events, face_matched_response_ids

def build_attention_events(stim_duration, inhibition_count):
    """Cree les periodes d'inhibition, randomisees sans chevauchement."""
    first_onset = PAUSE_DURATION + ATTENTION_MIN_STIM_OFFSET
    last_onset = PAUSE_DURATION + stim_duration - ATTENTION_MIN_STIM_OFFSET - ATTENTION_INHIBITION_DURATION
    if last_onset <= first_onset:
        raise ValueError("Intervalle trop court pour la tache d'attention centrale.")

    min_onset_spacing = (
        ATTENTION_MIN_ONSET_SPACING
        if inhibition_count == ATTENTION_INHIBITION_EVENTS
        else ATTENTION_INHIBITION_DURATION
    )

    required_span = (inhibition_count - 1) * min_onset_spacing
    available_span = last_onset - first_onset - required_span
    if available_span < -1e-9:
        raise RuntimeError("Impossible de placer les periodes d'inhibition avec l'espacement demande.")

    if available_span <= 1e-9:
        free_offsets = [0.0] * inhibition_count
    else:
        free_offsets = sorted(rng.uniform(0.0, available_span) for _ in range(inhibition_count))

    onsets = [
        first_onset + free_offsets[i] + (i * min_onset_spacing)
        for i in range(inhibition_count)
    ]

    events = []
    for onset in onsets:
        events.append({
            'event_type': 'inhibition',
            'onset': onset,
            'offset': onset + ATTENTION_INHIBITION_DURATION,
            'false_alarm_count': 0,
            'instruction_respected': True,
        })
    if len(events) != inhibition_count:
        raise RuntimeError(
            f"Nombre periodes inhibition incorrect : "
            f"{len(events)}/{inhibition_count}")
    return events

def build_attention_and_category_sequence(
        stim_duration, inhibition_count, total_events, detectable_face_count,
        inhibition_face_count):
    """Retire attention + visages jusqu'a obtenir un run respectant les quotas."""
    last_error = None
    for _ in range(1000):
        attention_events = build_attention_events(stim_duration, inhibition_count)
        if len(attention_events) != inhibition_count:
            last_error = RuntimeError(
                f"{inhibition_count} periodes attendues, "
                f"{len(attention_events)} generees.")
            continue
        try:
            category_sequence, planned_face_count, planned_inhibition_face_count = (
                build_category_sequence(
                    total_events, detectable_face_count, attention_events,
                    inhibition_face_count=inhibition_face_count,
                    stim_duration=stim_duration))
        except RuntimeError as exc:
            last_error = exc
            continue
        return (
            attention_events, category_sequence,
            planned_face_count, planned_inhibition_face_count)

    raise RuntimeError(
        "Impossible de generer un plan attention + visages respectant les "
        "quotas fixes.") from last_error

def classify_attention_events(attention_events, response_times, _face_matched_response_ids=None):
    false_alarm_total = 0
    respected_windows = 0
    for ev in attention_events:
        press_indices = [
            idx for idx, rt_abs in enumerate(response_times)
            if ev['onset'] <= rt_abs < ev['offset']
        ]
        ev['false_alarm_count'] = len(press_indices)
        ev['instruction_respected'] = (len(press_indices) == 0)
        false_alarm_total += len(press_indices)
        if ev['instruction_respected']:
            respected_windows += 1

    return {
        'windows_total': len(attention_events),
        'respected_windows': respected_windows,
        'false_alarms': false_alarm_total,
        'warning': false_alarm_total >= ATTENTION_WARNING_FALSE_ALARMS,
    }

def format_missed_frame_review(
        run_i, condition_name, missed_frames, frame_count,
        worst_frame_sec, face_events_count, object_events_count,
        stim_timing_summary, planned_face_count, response_summary,
        attention_summary, is_practice=False, run_label_override=None):
    timing_status = "OK" if stim_timing_summary['ok'] else "a verifier"
    run_label = run_label_override or ("Entrainement" if is_practice else f"Run {run_i}/{N_RUNS}")

    return (
        f"{run_label}\n\n"
        f"Visages a detecter : {response_summary['hits']}/{response_summary['planned_faces']} detectes"
        f" | visages inhibition : {response_summary['planned_inhibition_faces']}\n"
        f"Fausses alarmes : {response_summary['false_alarms']}\n"
        f"Cercle gris : {attention_summary['false_alarms']} appui(s) interdit(s)"
        f" | periodes OK : {attention_summary['respected_windows']}/{attention_summary['windows_total']}\n"
        f"Timing : {timing_status} | moyenne {stim_timing_summary['mean_hz']:.2f} Hz"
        f" | ecart max {stim_timing_summary['max_dev_ms']:.1f} ms"
        f" | attendu {stim_timing_summary['expected_ms']:.1f} ms\n\n"
        "ESPACE = SUIVANT")

def preview_folder(label, folder):
    """Montre une image aléatoire du dossier (vérification visuelle)."""
    paths = list_images(folder)
    if not paths:
        show_text(
            f"Preview : {label}\n\nDossier introuvable ou vide :\n{folder}"
            "\n\nSPACE = suivant",
            wait=True, with_fixation=False)
        return

    chosen = rng.choice(paths)
    img = visual.ImageStim(
        win=win, image=chosen,
        units='pix', size=(STIM_PIX, STIM_PIX), pos=(0, 0))
    title_stim = visual.TextStim(
        win=win, text=f"Preview : {label}",
        color='white', height=28, units='pix',
        pos=(0, SCREEN_RES[1] * 0.35))
    info_stim = visual.TextStim(
        win=win,
        text=f"{len(paths)} image(s) trouvée(s)\n{folder}\n\nSPACE = suivant | ESCAPE = quitter",
        color='white', height=18,
        wrapWidth=SCREEN_RES[0] * 0.9,
        units='pix', pos=(0, -SCREEN_RES[1] * 0.35))
    while True:
        title_stim.draw()
        img.draw()
        info_stim.draw()
        win.flip()
        keys = event.waitKeys(keyList=['space', 'escape'])
        if 'escape' in keys:
            clean_quit()
        if 'space' in keys:
            break

# %%
# ==============================
# Instructions et preview
# ==============================
intro = (
    "Analyse comportementale projet FVPS\n\n"
    "Gardez les yeux sur le point de fixation central durant toute la séquence.\n\n"
    "Placez le doigt qui vous convient le mieux sur la touche ESPACE.\n\n"
    "Lorsque le point de fixation central est un carre gris : appuyez sur la touche ESPACE si vous detectez un visage.\n"
    "Lorsque le point de fixation central est un cercle gris : n'appuyez pas sur la touche ESPACE, meme si vous detectez un visage.\n\n"
    "ESPACE = SUIVANT")
show_text(intro, wait=True, with_fixation=False)

if exp_info['Preview_stimuli'] == 'Yes':
    show_text("Preview des stimuli\n\nSPACE = voir les dossiers", wait=True)
    preview_folder(os.path.basename(FOLDER_ORI0_FACES),    FOLDER_ORI0_FACES)
    preview_folder(os.path.basename(FOLDER_ORI0_OBJECTS),  FOLDER_ORI0_OBJECTS)
    preview_folder(os.path.basename(FOLDER_ORI90_FACES),   FOLDER_ORI90_FACES)
    preview_folder(os.path.basename(FOLDER_ORI90_OBJECTS), FOLDER_ORI90_OBJECTS)

if DO_PRACTICE:
    show_text(
        "Debut des entrainements\n\n"
        "ESPACE = SUIVANT",
        wait=True, with_fixation=False)
else:
    show_text(
        "Debut de la tache principale\n\n"
        "ESPACE = SUIVANT",
        wait=True, with_fixation=False)

# %%
# ==============================
# Boucle principale : entrainements optionnels + 20 runs
# ==============================
event.clearEvents(eventType='keyboard')
kb.clock.reset()
timer.reset()

run_plan = []
if DO_PRACTICE:
    practice_sequence = ['HMstimuli90', 'HMstimuli0', 'VMstimuli90', 'VMstimuli0']
    rng.shuffle(practice_sequence)
    run_plan.extend({
        'condition': c,
        'is_practice': True,
        'detectable_face_count': None,
        'inhibition_face_count': 0
    } for c in practice_sequence)
for condition_name, detectable_face_count in zip(run_sequence, main_detectable_face_count_plan):
    run_plan.append({
        'condition': condition_name,
        'is_practice': False,
        'detectable_face_count': detectable_face_count,
        'inhibition_face_count': INHIBITION_FACE_RUN_COUNT
    })

main_run_i = 0
practice_run_i = 0
for run_item in run_plan:
    condition_name = run_item['condition']
    is_practice = run_item['is_practice']
    if is_practice:
        practice_run_i += 1
    else:
        main_run_i += 1
    run_i = main_run_i
    cfg = CONDITIONS[condition_name]
    ori = cfg['ori']
    p1_name, p2_name = cfg['pos1_name'], cfg['pos2_name']
    p1_pos,  p2_pos  = cfg['pos1'],      cfg['pos2']
    trig_code        = cfg['trigger_code']
    cond_code        = f"{cfg['axis'][0].upper()}{ori}"
    stim_events_this_run = PRACTICE_STIM_EVENTS_PER_RUN if is_practice else STIM_EVENTS_PER_RUN
    stim_duration_this_run = PRACTICE_STIM_DURATION if is_practice else MAIN_STIM_DURATION
    main_duration_this_run = PRACTICE_MAIN_DURATION if is_practice else MAIN_DURATION
    total_duration_this_run = PAUSE_DURATION + stim_duration_this_run
    total_frames_this_run = int(round(total_duration_this_run * REFRESH_RATE))
    detectable_face_count_this_run = (
        rng.choice(PRACTICE_FACE_RUN_COUNTS) if is_practice
        else run_item['detectable_face_count'])
    inhibition_face_count_this_run = run_item['inhibition_face_count']
    attention_inhibition_count = (
        PRACTICE_ATTENTION_INHIBITION_EVENTS if is_practice
        else ATTENTION_INHIBITION_EVENTS)

    # --- Pause entre runs ---
    run_label = f"Entrainement {practice_run_i}/{N_PRACTICE_RUNS}" if is_practice else f"Run {run_i}/{N_RUNS}"
    show_run_start(
        run_start_text(run_label, condition_name, stim_duration_this_run))

    # Réinitialiser l'état Échap
    esc_state['first_press_time'] = None
    esc_state['warning_shown']    = False

    # Variables de la séquence
    img_index       = 0
    current_p1      = None
    current_p2      = None
    current_face_p1 = False
    current_face_p2 = False
    current_onset   = 0.0
    current_stim_size = (STIM_PIX, STIM_PIX)
    response_times = []
    (
        attention_events, category_sequence,
        planned_face_count, planned_inhibition_face_count
    ) = build_attention_and_category_sequence(
        stim_duration_this_run, attention_inhibition_count,
        stim_events_this_run, detectable_face_count_this_run,
        inhibition_face_count_this_run)
    expected_face_events_this_run = planned_face_count + planned_inhibition_face_count
    if category_sequence.count('faces') != expected_face_events_this_run:
        raise RuntimeError(
            f"Plan visage invalide pour {run_label} : "
            f"{category_sequence.count('faces')}/{expected_face_events_this_run} visages.")
    image_decks = {
        'faces': build_image_deck(ori, 'faces'),
        'objects': build_image_deck(ori, 'objects')
    }
    run_face_events   = 0
    run_object_events = 0
    run_frame_count   = 0
    run_missed_frames = []
    run_worst_frame_sec = 0.0
    run_stim_onsets = []
    run_frame_duration_ms_values = []

    # Historique des onsets face pour les classifications
    # Structure : {onset: {face_present, responded, rt, classified}}
    face_events = {}

    seq_clock    = core.Clock()
    last_t       = None
    quit_flag    = False

    event.clearEvents(eventType='keyboard')
    kb.clock.reset()
    seq_clock.reset()

    for frame_n in range(total_frames_this_run):
        t_seq = frame_n / REFRESH_RATE
        now   = seq_clock.getTime()

        # --- Phases ---
        if t_seq < PAUSE_DURATION:
            show_images = False
            fade        = 0.0
            phase       = 'pause'
        elif t_seq < PAUSE_DURATION + FADE_DURATION:
            show_images = True
            fade        = (t_seq - PAUSE_DURATION) / FADE_DURATION
            phase       = 'fade_in'
        elif t_seq < PAUSE_DURATION + FADE_DURATION + main_duration_this_run:
            show_images = True
            fade        = 1.0
            phase       = 'main'
        else:
            show_images = True
            fp          = (t_seq - PAUSE_DURATION - FADE_DURATION - main_duration_this_run) / FADE_DURATION
            fade        = max(0.0, 1.0 - fp)
            phase       = 'fade_out'

        # --- Mesure missed frames ---
        if last_t is not None:
            frame_dur = now - last_t
            missed    = 1 if frame_dur > MISSED_FRAME_THRESHOLD else 0
            if not is_practice:
                run_frame_duration_ms_values.append(frame_dur * 1000.0)
            run_frame_count += 1
            run_worst_frame_sec = max(run_worst_frame_sec, frame_dur)
            if missed:
                run_missed_frames.append((frame_n, t_seq, frame_dur))
        last_t = now

        # --- Nouveau stimulus duplique aux deux positions a 6 Hz ---
        if show_images and frame_n % FrameN == 0:
            if img_index < len(category_sequence):
                category = category_sequence[img_index]
            else:
                category = 'objects'
            path_stim = draw_image(image_decks, ori, category)
            current_onset = seq_clock.getTime()
            run_stim_onsets.append(current_onset)
            if category == 'faces':
                run_face_events += 1
            else:
                run_object_events += 1

            current_p1      = path_stim
            current_p2      = path_stim
            current_stim_size, _current_stim_scale = draw_stim_size()
            current_face_p1 = (category == 'faces')
            current_face_p2 = (category == 'faces')
            face_present    = current_face_p1 or current_face_p2
            scored_for_face_task = not is_in_attention_window(current_onset, attention_events)
            fixation_condition = FIXATION_SQUARE_LABEL if scored_for_face_task else FIXATION_CIRCLE_LABEL
            time_in_stimulation = max(0.0, t_seq - PAUSE_DURATION)
            miss_exempt_due_to_fade = (
                face_present
                and is_fade_miss_exempt(time_in_stimulation, stim_duration_this_run)
            )

            # Enregistrement de l'événement face pour classification ultérieure
            face_events[current_onset] = {
                'img_index':  img_index,
                'cat_p1':     category,
                'cat_p2':     category,
                'face_present': face_present,
                'scored_for_face_task': scored_for_face_task,
                'miss_exempt_due_to_fade': miss_exempt_due_to_fade,
                'fixation_condition': fixation_condition,
                'image_p1':   safe_filename(path_stim),
                'image_p2':   safe_filename(path_stim),
                'responded':  False,
                'rt':         None,
                'classification': None,
                'written':    False,
            }
            img_index += 1

        # --- Lecture des reponses : SPACE uniquement ---
        new_keys = kb.getKeys(keyList=['space'], clear=True)
        for kp in new_keys:
            if kp.name == 'space':
                response_times.append(kp.rt)

        # --- Dessin ---
        if show_images:
            final_contrast = fade * sinusoidal_stim_contrast(frame_n)

            if current_p1 is not None:
                s1 = image_cache[current_p1]
                s1.pos      = p1_pos
                s1.size     = current_stim_size
                s1.contrast = final_contrast
                s1.draw()

            if current_p2 is not None:
                s2 = image_cache[current_p2]
                s2.pos      = p2_pos
                s2.size     = current_stim_size
                s2.contrast = final_contrast
                s2.draw()

        # Point de fixation et periodes d'inhibition toujours visibles.
        draw_fixation_marker(t_seq, attention_events)

        # Avertissement Échap si premier appui actif
        if esc_warning_active():
            esc_warning_stim.draw()

        win.flip()

        # --- Gestion Échap ---
        if check_escape():
            quit_flag = True
            break

    if not quit_flag:
        response_clock = core.Clock()
        while response_clock.getTime() < RESPONSE_WINDOW_SEC:
            # Meme touche pendant la fenetre de reponse finale.
            new_keys = kb.getKeys(keyList=['space'], clear=True)
            for kp in new_keys:
                if kp.name == 'space':
                    response_times.append(kp.rt)
            win.flip()
            if check_escape():
                quit_flag = True
                break

    face_events, face_matched_response_ids = classify_events(face_events, response_times, attention_events)
    attention_summary = classify_attention_events(attention_events, response_times, face_matched_response_ids)
    response_summary = {
        'hits': sum(1 for ev in face_events.values() if ev['classification'] == 'hit'),
        'misses': sum(1 for ev in face_events.values() if ev['classification'] == 'miss'),
        'false_alarms': sum(1 for ev in face_events.values() if ev['classification'] == 'false_alarm'),
        # Pourcentage de detection : hits et misses vraiment scores.
        'faces': sum(1 for ev in face_events.values()
                     if ev['classification'] in ('hit', 'miss')),
        'planned_faces': planned_face_count,
        'planned_inhibition_faces': planned_inhibition_face_count,
        'presented_faces': run_face_events
    }
    if not is_practice:
        for ons in sorted(face_events):
            ev = face_events[ons]
            stimulus_category = 'face' if ev['cat_p1'] == 'faces' else 'object'
            append_main_row([
                f"{participant}_{session_dt}", logged_condition_name(condition_name),
                *condition_excel_fields(condition_name),
                ev['fixation_condition'],
                stimulus_category, round(ons, 4),
                'space' if ev['responded'] else 'none',
                ev['rt'] if ev['rt'] is not None else '', ev['classification']])

    if quit_flag:
        break

    if run_face_events != expected_face_events_this_run:
        raise RuntimeError(
            f"Nombre de visages presentes invalide pour {run_label} : "
            f"{run_face_events}/{expected_face_events_this_run}.")

    stim_timing_summary = summarize_stim_timing(run_stim_onsets)
    if not is_practice:
        append_frame_summary_row(
            condition_name, run_frame_duration_ms_values, stim_timing_summary)
    event.clearEvents(eventType='keyboard')
    review_text = format_missed_frame_review(
        run_i, condition_name, run_missed_frames, run_frame_count,
        run_worst_frame_sec, run_face_events, run_object_events,
        stim_timing_summary, planned_face_count, response_summary,
        attention_summary, is_practice=is_practice, run_label_override=run_label)
    detection_feedback = format_face_detection_percent(response_summary)
    attention_warning = format_attention_warning(attention_summary['false_alarms'])
    show_run_start(review_text, warn_fixation=attention_summary['warning'],
                   highlight=detection_feedback, warning_text=attention_warning)
    if is_practice and practice_run_i == N_PRACTICE_RUNS:
        show_text(
            "Fin des entrainements\n\n"
            "L'experience principale va commencer.\n"
            "Gardez les yeux sur le point de fixation central.\n\n"
            "ESPACE = SUIVANT",
            wait=True, with_fixation=False)
    kb.clock.reset()

# %%
# ==============================
# Fin de l'expérience
# ==============================
main_xlsx_saved, frames_xlsx_saved = save_workbooks()
xlsx_message = (
    f"\n\nTableau donnees participant Excel :\n{out_xlsx_fname}"
    if main_xlsx_saved else
    "\n\nExport Excel donnees participant impossible : voir la console.")
xlsx_message += (
    f"\n\nTableau frames Excel :\n{log_xlsx_fname}"
    if frames_xlsx_saved else
    "\n\nExport Excel frames impossible : voir la console.")
show_text(
    f"Fin de l'experience.\n\nMerci !\n\n"
    f"Conditions : filtrees {FILTER_DEGREE}\n"
    f"Runs principaux : {N_RUNS} sequences de {MAIN_STIM_DURATION:.0f} s\n\n"
    f"Donnees comportementales sauvegardees dans :\n{MAIN_DATA_DIR}"
    f"{xlsx_message}",
    wait=False, with_fixation=False)
core.wait(3)
win.close()
core.quit()
