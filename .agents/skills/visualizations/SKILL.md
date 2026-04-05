# YouPlot (uplot) — Terminal Unicode Plotting

> **Source**: <https://github.com/red-data-tools/YouPlot> (v0.4.6, MIT)
> Powered by [UnicodePlot](https://github.com/red-data-tools/unicode_plot.rb)

## What is YouPlot?

YouPlot (`uplot` / `youplot`) is a Ruby CLI that draws Unicode bar charts, histograms, line plots, scatter plots, density plots, and box plots directly in the terminal. It reads delimited columnar data from **stdin** or **files** and renders plots to stderr (by default), making it composable in Unix pipelines.

---

## Installation

```bash
# macOS (Homebrew)
brew install youplot

# Any platform with Ruby ≥ 2.5
gem install youplot

# Nix
nix shell nixpkgs#youplot

# Conda (install Ruby first)
conda install -c conda-forge ruby compilers && gem install youplot
```

Binary names: **`uplot`** (short) and **`youplot`** (long). They are identical.

---

## General Invocation

```
# stdin
cat data.tsv | uplot <subcommand> [options]

# file arguments
uplot <subcommand> [options] file1.tsv file2.tsv ...

# passthrough pipeline (data forwarded to stdout)
pipeline1 | uplot <subcommand> -O | pipeline2
```

---

## Subcommands

| Subcommand   | Short | Description                                      |
|-------------|-------|--------------------------------------------------|
| `barplot`   | `bar` | Horizontal bar chart                             |
| `histogram` | `hist`| Horizontal histogram (continuous data)           |
| `lineplot`  | `line`| Single-series line chart                         |
| `lineplots` | `lines`| Multi-series line chart                         |
| `scatter`   | `s`   | Scatter plot (2-D or multi-series)               |
| `density`   | `d`   | Density plot (scatter-like with density shading) |
| `boxplot`   | `box` | Horizontal box-and-whisker plot                  |
| `count`     | `c`   | Bar chart of value frequencies (slow on large N) |
| `colors`    | `color`| Print the list of available named colors        |

---

## Global Options (apply to all subcommands)

| Flag                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `-o [FILE]`           | Output **plot** to file or stdout (`-o -` or bare `-o` → stdout)            |
| `-O [FILE]`           | Pass **input data** through to stdout (`-O -` or bare `-O` → stdout)        |
| `-d DELIM`            | Field delimiter (default: **tab** `\t`). Use `-d,` for CSV, `-d ' '` for space |
| `-H`                  | First row is a header (used for series labels / title)                      |
| `-t TITLE`            | Plot title string                                                           |
| `-w WIDTH`            | Canvas width in characters                                                  |
| `-h HEIGHT`           | Canvas height in characters                                                 |
| `-b BORDER`           | Border style                                                                |
| `-m MARGIN`           | Left margin                                                                 |
| `-p PADDING`          | Padding                                                                     |
| `-c COLOR`            | Color name (run `uplot colors` to list). Not all subcommands support this.  |
| `--xlim MIN,MAX`      | X-axis limits (comma-separated)                                            |
| `--ylim MIN,MAX`      | Y-axis limits (comma-separated)                                            |
| `--xlabel LABEL`      | X-axis label                                                                |
| `--ylabel LABEL`      | Y-axis label                                                                |
| `--fmt FORMAT`        | Column format: `xyy` (default), `xyxy`, `yx`                               |
| `-T`                  | Transpose the data                                                          |
| `--progress` / `-p`   | Experimental progressive/real-time mode                                     |
| `--config`            | Show config file path (youplotrc, YAML)                                     |
| `--help`              | Detailed help for the subcommand                                            |
| `--version`           | Print version                                                               |

### Subcommand-specific options

| Flag             | Subcommand | Description                        |
|-----------------|------------|------------------------------------|
| `--nbins N`     | `hist`     | Number of histogram bins           |
| `--closed SIDE` | `hist`     | Closed side of bins: `left`/`right`|
| `--symbol CHAR` | `bar`      | Character for bars                 |
| `--xscale SCALE`| `bar`      | X-axis scale (`log`, etc.)         |
| `--reverse`     | `count`    | Reverse sort order for count       |

> **Tip**: Always run `uplot <subcommand> --help` for the authoritative option list of that subcommand.

---

## Column / Axis Conventions

- **Default**: Column 1 = X, Column 2 = Y. With multiple series: Col1 = X, Col2 = Y1, Col3 = Y2, …
- **Single column** for `line` or `bar`: auto-generates sequential X (1, 2, 3, …).
- `--fmt xyy`  — (default) first col is shared X, remaining cols are Y series.
- `--fmt xyxy` — pairs of (X, Y) columns: Col1=X1, Col2=Y1, Col3=X2, Col4=Y2, …
- `--fmt yx`   — first col is Y (labels), second col is X (values). Useful with `sort | uniq -c` output.
- Swap columns: `awk '{print $2, $1}'`
- Concatenate series: `paste file1.tsv file2.tsv`

---

## Recipes & Patterns

### Bar chart from CSV

```bash
cat data.csv | sort -nk2 -t, | tail -n15 | uplot bar -d, -t "Top 15"
```

### File sizes bar chart (no external data)

```bash
ls -l | awk '{print $9, $5}' | sort -nk 2 | uplot bar -d ' '
```

### Histogram from Python-generated data

```bash
python3 -c "
import numpy as np
for x in np.random.randn(10000): print(x)
" | uplot hist --nbins 20
```

### Line plot with axis limits

```bash
cat timeseries.csv | cut -f2,3 -d, \
| uplot line -d, -w 50 -h 15 -t "Air Passengers" --xlim 1950,1960 --ylim 0,600
```

### Sine wave (generated inline)

```bash
python3 -c '
from math import sin, pi
for i in range(101):
    x = i * pi / 50
    print(f"{x}\t{sin(x)}")
' | uplot line
```

### Multi-column scatter with header

```bash
cat iris.tsv | cut -f1-4 | uplot scatter -H -t IRIS
```

### Density plot

```bash
cat iris.csv | cut -f1-4 -d, | uplot density -H -d, -t IRIS
```

### Box plot

```bash
cat iris.csv | cut -f1-4 -d, | uplot boxplot -H -d, -t IRIS
```

### Count / frequency bar chart

```bash
# Count occurrences (slow Ruby implementation)
ps aux | awk '{print $1}' | uplot count

# Faster alternative using Unix tools + uplot bar
cat data.txt | sort | uniq -c | sort -nrk1 \
| uplot bar --fmt yx -d ' ' -t "Frequency"
```

### Passthrough pipeline (plot + forward data)

```bash
generate_data | uplot line -O | downstream_consumer
```

### Chromosome gene count (bioinformatics)

```bash
cat gencode.v35.annotation.gff3 \
| grep -v '#' | grep 'gene' | cut -f1 \
| sort | uniq -c | sort -nrk1 \
| uplot bar --fmt yx -d ' ' -t "Genes per chromosome" -c blue
```

### Colored output

```bash
# List available colors
uplot colors

# Use a named color
echo "..." | uplot bar -c cyan
```

### Real-time / streaming plot (experimental)

```bash
ruby -e 'loop { puts rand(100) }' | uplot line --progress
```

---

## Configuration File (youplotrc)

YouPlot supports a YAML config for default options. Run `uplot --config` to see the path and format. Example:

```yaml
# ~/.youplotrc (example)
width: 60
height: 20
border: :ascii
```

---

## Companion Tools for Data Wrangling

These tools pair well with `uplot` for preprocessing columnar data:

| Tool           | Purpose                        | Example                                        |
|---------------|--------------------------------|------------------------------------------------|
| `awk`         | Column selection/reordering    | `awk '{print $2, $1}'`                         |
| `cut`         | Column slicing                 | `cut -f1-4 -d,`                                |
| `sort`        | Sorting rows                   | `sort -nk2 -t,`                                |
| `uniq -c`     | Counting unique values         | `sort \| uniq -c`                              |
| `paste`       | Merging columns from files     | `paste x.tsv y.tsv`                            |
| `sed`         | Stream editing / cleanup       | `sed '/^$/d'`                                  |
| `csvtk`       | CSV/TSV Swiss Army knife       | `csvtk cut -f1,3`                              |
| `xsv`         | Fast CSV toolkit (Rust)        | `xsv select 1,3`                               |
| `GNU datamash`| Grouping / aggregation         | `datamash -g1 mean 2`                          |
| `tail` / `head`| Row slicing                  | `tail -n15`                                    |

---

## Gotchas & Tips

1. **Delimiter default is TAB** — for CSV use `-d,`.
2. **Header flag `-H`** is required when the first row has labels; without it, parsing will fail or produce garbage.
3. **Plot goes to stderr** — redirect with `-o` if you need it in stdout or a file.
4. **`count` is slow** on large datasets — prefer `sort | uniq -c | uplot bar --fmt yx` for speed.
5. **Line colors by number** are not supported — use named colors only (see `uplot colors`).
6. **Time series** are not natively supported — convert timestamps to numeric epoch or ordinal before plotting.
7. **Categorical scatter** requires GNU datamash gymnastics — see the README for the datamash + `--fmt xyxy` pattern.
8. **Multiple files** can be passed as positional arguments: `uplot line file1.tsv file2.tsv`.
9. **Terminal width** affects rendering — use `-w` and `-h` to control canvas size explicitly.

---

## Quick Reference Card

```
# Bar chart from CSV
cat f.csv | uplot bar -d, -t "Title"

# Histogram (20 bins)
cat vals.txt | uplot hist --nbins 20

# Line plot (sized)
cat xy.tsv | uplot line -w 60 -h 20

# Multi-series lines
cat multi.tsv | uplot lines -H

# Scatter
cat xy.tsv | uplot s -H -t "My Scatter"

# Density
cat xy.tsv | uplot d -H

# Box plot
cat cols.tsv | uplot box -H

# Frequency count (fast)
sort data.txt | uniq -c | sort -nrk1 | uplot bar --fmt yx -d ' '

# Passthrough
gen | uplot line -O | consume

# Colors
uplot colors
```
