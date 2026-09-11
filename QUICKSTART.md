# Quick start

Get fmri-gym running and play every currently supported DBP game with a
one-liner. For design notes, adapters, and logging details see [README.md](README.md).

## 1. Install

This checkout uses **uv** with a `.venv/` in the repo root. Run every install
command from the repo root: `uv pip` finds `./.venv` on its own, so you do not
have to activate anything first.

```bash
uv venv --python 3.11           # creates .venv/
uv pip install -r requirements.txt
```

To run the scripts, either activate the env (`source .venv/bin/activate`) or
call the interpreter directly (`.venv/bin/python fmri_play.py ...`). Note that a
uv-created venv has no `pip` inside it — use `uv pip install ...`, not
`.venv/bin/pip`.

<details>
<summary>conda instead of uv</summary>

```bash
conda create -n fmri-gym python=3.11
conda activate fmri-gym
pip install -r requirements.txt
```

Everything below works the same; just read `uv pip install` as `pip install`.
</details>

`requirements.txt` already pulls in the common backends (crafter, minihack,
vizdoom, playwright, pystk2-gymnasium, …). Two games need an extra step:

```bash
# Baba is AI
uv pip install "git+https://github.com/nacloos/baba-is-ai.git"

# Rush Hour — Python package + Go engine binary
uv pip install "git+https://github.com/chrplr/Rush-Hour.git#subdirectory=python"
git clone https://github.com/chrplr/Rush-Hour vendor/rush-hour-src
# Build the binary in place; leave it at vendor/rush-hour-src/rushhour-env
# (the adapter looks there automatically — do not move it):
cd vendor/rush-hour-src && go build -o rushhour-env ./cmd/rushhour-env && cd ../..
# Optional: if you built it elsewhere, point at it with:
#   export RUSHHOUR_ENV_BIN=/absolute/path/to/rushhour-env
```

If you already have a Rush-Hour checkout, install it editable and symlink the
binary rather than cloning a second copy:

```bash
RH=/path/to/Rush-Hour
uv pip install -e "$RH/python"
(cd "$RH" && go build -o rushhour-env ./cmd/rushhour-env)
mkdir -p vendor/rush-hour-src        # a real directory, so .gitignore covers it
ln -s "$RH/rushhour-env" vendor/rush-hour-src/rushhour-env
```

The default Go build is headless (no SDL, no C toolchain), so `go build` needs
nothing but the Go compiler.

AI GameStore uses Playwright + system Chrome by default. If you don't have
Chrome, install the bundled Chromium instead:

```bash
playwright install chromium   # then set "browser_channel": null in the phase if needed
```

## 2. How a session works

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/<game>.json
```

| Flag / key | What it does |
|---|---|
| `--subject sub-01` | Subject id used in the output folder name |
| **SPACE** | Advance past the experimenter screen |
| **`=`** | Scanner trigger (anchors the session clock) |
| **ESC** | Quit early; data is still saved |

Each config is a short curriculum: message → fixation → game (~300 s, auto-restarts
on game-over) → fixation. Output lands in `data/<subject>_<timestamp>/`.

Runtime: experimenter screen (**SPACE**) → "Waiting for scanner..." → trigger **`=`** → curriculum.

## 3. Run every game

All commands assume you're in the repo root with `fmri-gym` activated.

### ViZDoom

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/vizdoom__defend_center.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/vizdoom__deadly_corridor.json
```

Controls: arrows move/turn, Z/X strafe, SPACE shoots.

### Crafter

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/crafter__crafter.json
```

### Rush Hour

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/rushhour__easy.json      # 5 min of random easy puzzles
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/rushhour_complete.json    # Rush-Hour's own session: 12 puzzles, easiest first, self-paced
```

`rushhour_complete.json` reproduces the flow of the Rush-Hour program itself
(ready screen before each puzzle, blank ITI, solved hold, its look and its log
columns) inside fmri-gym; see the `_session_note` in the file for what is not
reproduced (SDL-clock timestamps).

Needs the Go binary from §1. After the build step you should have
`vendor/rush-hour-src/rushhour-env` in the repo — leave it there; the adapter
auto-finds that path. If the binary lives somewhere else, set
`export RUSHHOUR_ENV_BIN=/absolute/path/to/rushhour-env` before running.

### Baba is AI

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/baba__make_win.json
```

### AI GameStore (p5.js browser games)

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game1.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game2.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game3.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game4.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game5.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game6.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game7.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game8.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game9.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game10.json
```

Controls: arrows + SPACE / Z / ENTER (game-dependent). Needs Playwright + Chrome (§1).

### MiniHack

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__room5x5.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__room15x15.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__mazewalk9x9.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__river.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__corridor.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__eat.json
```

Controls: arrow keys (N/E/S/W). Needs `setuptools<81` (already in `requirements.txt`).

### SuperTuxKart

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/supertuxkart__race.json
```

Needs a real GL display (does **not** work under `SDL_VIDEODRIVER=dummy`).
Controls: arrows steer/accelerate/brake, SPACE fire, Z drift, X nitro.

## Tips

- Useful flags: `--size 1280x1024`, `--fullscreen`.
- Archived / unsupported configs live under `configs/dbp_games/archive/` and
  `configs/dbp_games/unsupported/` — see the README for the wider game list.
- Per-config `_note` / `_game` fields document setup quirks for that title.
