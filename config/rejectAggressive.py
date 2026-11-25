# Aggressive outlier (cosmic-ray) rejection in the extraction of binned data, for ReduceExposureTask.
#
# Misses few cosmic rays, at the cost of rejecting many more good pixels (especially where the spectrum
# changes rapidly from row to row, e.g. near the dichroic and in the 930-950nm sky-line forest). Suitable when
# only the summed flux of a spectrum matters (e.g. photometry of bright stars down the fibre): rejected
# fiber-rows become NO_DATA, and interpolating over them costs ~0.01% of the summed flux per row.
#
# Use with: pipetask run ... -C reduceExposure:$DRP_STELLA_DIR/config/rejectAggressive.py
# or the pipeline $DRP_STELLA_DIR/pipelines/scienceAggressiveCR.yaml
config.extractSpectra.rejectThreshold = 5.0
config.extractSpectra.rejectSystematicFraction = 0.02
config.extractSpectra.rejectSystematicRows = 0
config.extractSpectra.rejectBadFitThreshold = 5.0  # chi^2/dof > 3 happens by chance in ~10-pixel fiber-rows
config.extractSpectra.rejectGapThreshold = 5.0
config.extractSpectra.rejectGrowThreshold = 2.0
config.extractSpectra.rejectGrowAcrossRowsThreshold = 3.0
