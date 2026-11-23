# LSF investigations

Claude: the scripts behind the line-spread function work of PIPE2D-1940 (`ArmLsf`, the LSF stored in the
spectra) and PIPE2D-1993 (the LSF of pfsMerged and pfsCoadd). They are kept as a record of how the results
in the tickets and in the LSF chapter of the documentation were measured, and as starting points for the
follow-ups. They are not maintained tools: those are in `bin` (`pfsLsfResiduals.py`, `pfsLsfWidths.py`,
`pfsCheckArmLsf.py`, `pfsLsfChi2.py`, `pfsCompareRuns.py`) and
`pfs.drp.stella.lsfDiagnostics`.

The Butler repository is `$PFS_REPO` (default `/work/datastore`). Reading `pfsArm.lsf` or `pfsMerged.lsf`
needs the repository's `butler.yaml` to use obs_pfs's `PfsFiberArraySetFormatter` for PfsArm and PfsMerged;
until it does, set `PFS_REPO` to a copy of the `butler.yaml` that does. The default collections are the ones
used in the tickets: the armLsf fit to the Run 30 arcs is `u/rhl/PIPE2D-1940-armLsf-lampAvg`, and the
PIPE2D-1993 regression used the `u/kiyoyabe/weekly_hilo/w.2026.04` weekly. Those collections were written
before the LSF model became a calibration, so the scripts read it as `armLsf`; models fit since are
`armLsf_calib`.

## Fitting the LSF of a detector

| Script | What it does |
|---|---|
| `fitLsfOneDetector.py` | Runs `FitLsfTask` on one detector without pipetask, writes `armLsf` and `lsfLines` to files, and uses the model (the example in the LSF chapter of the documentation) |

## Why do the lamps' line widths differ? (PIPE2D-1940)

The model's per-visit offsets average zero over the lamps; these scripts test what the offsets are.

| Script | Question |
|---|---|
| `lsfLinesByLamp.py` | Line widths relative to the model, by visit (lamp) |
| `lampDiffChecks.py` | Do the lamps differ in the same way across the detector, and with S/N? |
| `lampLineWidths.py` | Are the offsets per lamp, or per line? |
| `lineBlendTest.py` | Are the broad lines blends? (excess width against catalogue neighbours) |
| `lampPupilChecks.py` | Do the lamps fill the telescope pupil differently? (rms and kurtosis offsets together) |
| `lampRepeats.py` | Do line widths drift between repeated arcs of one lamp? |

## The LSF stored with the spectra (PIPE2D-1940)

| Script | Question |
|---|---|
| `checkStoredLsf.py` | Do `pfsMerged.lsf` and `pfsCoadd.lsf` agree with the deprecated pickles? |

## The LSF of combined spectra (PIPE2D-1993)

| Script | Question |
|---|---|
| `combinedLsfWidths.py` | How much wider is `CombinedLsf` than the old `CoaddLsf`? (plots) |
| `timeCombinedLsf.py` | How long does mergeArms take to make the `CombinedLsf`? |
| `perPixelVsSmooth.py` | How far is the smooth-weight LSF from the one implied by the per-pixel flux weights? (PIPE2D-2006) |
| `resampleCovariance.py` | How does resampling white noise beat the diagonal variance, and what does ignoring the covariance cost? |
| `fluxCalSpikes.py` | Unmasked spikes in the fluxCal of the regression visits (PIPE2D-2005) |
