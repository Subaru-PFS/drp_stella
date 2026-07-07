import numpy as np
import pandas as pd
from pfs.utils.coordinates.Subaru_POPT2_PFS import PFS

__all__ = [
    'ag_pfimm_to_zenith_offset',
    'fit_global_model',
    'fit_global_model_pfimm',
]


def ag_pfimm_to_zenith_offset(xpfi, ypfi, inr):
    """Convert PFI mm positions to offsets from boresight in a zenith-aligned frame.

    Parameters
    ----------
    xpfi : float, array, or Series
        x position on PFI (mm)
    ypfi : float, array, or Series
        y position on PFI (mm), in the ag_pixel_to_pfimm sign convention
    inr : float
        Instrument rotator angle in degrees

    Returns
    -------
    dz : float or array
        Offset towards zenith (mm on telescope focal plane).
        Positive = towards zenith.
    dp : float or array
        Offset perpendicular to zenith (mm on telescope focal plane).
        Positive = towards the Opt Nasmyth platform side (+xfp).

    Notes
    -----
    The telescope focal plane (fp) has:
        +yfp = "Rear" direction = towards zenith for an alt-az telescope
        +xfp = "Opt" side = perpendicular to zenith (horizontal)

    The plate scale at the AG camera radius is ~11 arcsec/mm.

    The inputs must be in the ag_pixel_to_pfimm convention (ypfi has
    pfi_parity applied).  When using agc_nominal_x_mm / agc_nominal_y_mm
    from the opDB (read with flipToHardwareCoords=True), you must negate
    y because flipToHardwareCoords produces the det2dp/fp2pfi sign
    convention, which is opposite to ag_pixel_to_pfimm in y::

        ag_pfimm_to_zenith_offset(agcData.agc_nominal_x_mm,
                                  -agcData.agc_nominal_y_mm, insrot)
    """
    if isinstance(xpfi, pd.Series):
        xpfi = xpfi.to_numpy()
    if isinstance(ypfi, pd.Series):
        ypfi = ypfi.to_numpy()

    xpfi = np.atleast_1d(np.asarray(xpfi, dtype=float))
    ypfi = np.atleast_1d(np.asarray(ypfi, dtype=float))

    pfs = PFS()
    xfp, yfp = pfs.pfi2fp(xpfi, ypfi, inr)

    dz = yfp
    dp = xfp

    return dz, dp


def fit_global_model(dz_nominal, dp_nominal, dz_centroid, dp_centroid,
                     agc_exposure_id, agc_camera_id=None, valid=None,
                     fitPfiRotation=True, fitPfiScale=False,
                     fitAgcOffsets=False, fitAgcRotation=False):
    """Fit a global boresight model per exposure and return the model prediction.

    Uses an iterative alternating approach: fit per-exposure terms (subtracting
    current camera model), then fit per-camera terms (subtracting current
    exposure model), repeating for 3 iterations.

    The fit is done in the zenith-aligned frame (dz towards zenith, dp
    perpendicular), so per-camera offsets are constant even under gravity flexure.

    Parameters
    ----------
    dz_nominal : array or Series
        Nominal zenith-direction position (mm).
    dp_nominal : array or Series
        Nominal perpendicular position (mm).
    dz_centroid : array or Series
        Centroid zenith-direction position (mm).
    dp_centroid : array or Series
        Centroid perpendicular position (mm).
    agc_exposure_id : array or Series
        Exposure identifier.  Fit is done per exposure across all cameras.
    agc_camera_id : array or Series, optional
        Camera identifier [0, 5].  Required if fitAgcOffsets is True.
    valid : array of bool, optional
        Mask of stars to use in the fit (e.g. agc_match_flags == 1).
        If None, all stars are used.
    fitPfiRotation : bool
        If True, fit a per-exposure rotation of the PFI about the boresight.
    fitPfiScale : bool
        If True, fit a per-exposure scale change of the PFI.
    fitAgcOffsets : bool
        If True, fit a constant (dz, dp) offset per camera across all exposures.
    fitAgcRotation : bool
        If True, also fit a constant rotation per camera (about that camera's
        centre).  Only used if fitAgcOffsets is True.

    Returns
    -------
    model_dz : ndarray
        Model displacement towards zenith (mm).
    model_dp : ndarray
        Model displacement perpendicular to zenith (mm).
    params : dict
        'exposure': DataFrame indexed by agc_exposure_id with columns
            dalt (mm), daz (mm), rotation (arcsec), scale (dimensionless).
            Unfitted parameters are set to 0.
        'camera': DataFrame indexed by agc_camera_id with columns
            dalt (mm), daz (mm), rotation (arcsec).
            Unfitted parameters are set to 0.  Only present if fitAgcOffsets is True.
    """
    dz_nom = np.asarray(dz_nominal, dtype=float)
    dp_nom = np.asarray(dp_nominal, dtype=float)
    dz_cen = np.asarray(dz_centroid, dtype=float)
    dp_cen = np.asarray(dp_centroid, dtype=float)
    exp = np.asarray(agc_exposure_id)

    if valid is None:
        use = np.ones(len(dz_nom), dtype=bool)
    else:
        use = np.asarray(valid, dtype=bool)

    if fitAgcOffsets:
        if agc_camera_id is None:
            raise ValueError("agc_camera_id required when fitAgcOffsets=True")
        cam = np.asarray(agc_camera_id)
        unique_cam = np.unique(cam)

    err_dz = dz_cen - dz_nom
    err_dp = dp_cen - dp_nom

    N = len(dz_nom)
    unique_exp = np.unique(exp)

    # Iterative approach: alternate per-exposure and per-camera fits
    cam_model_dz = np.zeros(N)
    cam_model_dp = np.zeros(N)

    niter = 3 if fitAgcOffsets else 1
    for iteration in range(niter):
        # --- Per-exposure fit (subtract current camera model) ---
        resid_dz = err_dz - cam_model_dz
        resid_dp = err_dp - cam_model_dp

        exp_model_dz = np.zeros(N)
        exp_model_dp = np.zeros(N)
        exp_records = []

        for eid in unique_exp:
            sel = exp == eid
            n = np.sum(sel)
            if n < 2:
                exp_records.append({'agc_exposure_id': eid,
                                    'dalt': 0.0, 'daz': 0.0,
                                    'rotation': 0.0, 'scale': 0.0})
                continue

            fit_mask = use[sel]
            nfit = np.sum(fit_mask)
            if nfit < 2:
                exp_records.append({'agc_exposure_id': eid,
                                    'dalt': 0.0, 'daz': 0.0,
                                    'rotation': 0.0, 'scale': 0.0})
                continue

            ez = resid_dz[sel][fit_mask]
            ep = resid_dp[sel][fit_mask]
            nz = dz_nom[sel][fit_mask]
            np_ = dp_nom[sel][fit_mask]

            ncols = 2 + fitPfiRotation + fitPfiScale
            A = np.zeros((2 * nfit, ncols))
            A[:nfit, 0] = 1.0
            A[nfit:, 1] = 1.0
            c = 2
            if fitPfiRotation:
                A[:nfit, c] = -np_
                A[nfit:, c] = nz
                c += 1
            if fitPfiScale:
                A[:nfit, c] = nz
                A[nfit:, c] = np_

            b = np.concatenate([ez, ep])
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            rec = {'agc_exposure_id': eid, 'dalt': x[0], 'daz': x[1]}
            nz_all = dz_nom[sel]
            np_all = dp_nom[sel]
            mdz = x[0] * np.ones(n)
            mdp = x[1] * np.ones(n)
            c = 2
            if fitPfiRotation:
                rec['rotation'] = np.rad2deg(x[c]) * 3600.0
                mdz += x[c] * (-np_all)
                mdp += x[c] * nz_all
                c += 1
            else:
                rec['rotation'] = 0.0
            if fitPfiScale:
                rec['scale'] = x[c]
                mdz += x[c] * nz_all
                mdp += x[c] * np_all
            else:
                rec['scale'] = 0.0

            exp_model_dz[sel] = mdz
            exp_model_dp[sel] = mdp
            exp_records.append(rec)

        if not fitAgcOffsets:
            break

        # --- Per-camera fit (subtract current exposure model) ---
        resid_dz = err_dz - exp_model_dz
        resid_dp = err_dp - exp_model_dp

        cam_model_dz = np.zeros(N)
        cam_model_dp = np.zeros(N)
        cam_records = []

        for cid in unique_cam:
            sel = cam == cid
            fit_mask = sel & use
            nfit = np.sum(fit_mask)
            if nfit < 2:
                cam_records.append({'agc_camera_id': int(cid),
                                    'dalt': 0.0, 'daz': 0.0,
                                    'rotation': 0.0})
                continue

            ez = resid_dz[fit_mask]
            ep = resid_dp[fit_mask]

            if fitAgcRotation:
                nz_c = dz_nom[fit_mask]
                np_c = dp_nom[fit_mask]
                nz_mean = np.mean(nz_c)
                np_mean = np.mean(np_c)

                A = np.zeros((2 * nfit, 3))
                A[:nfit, 0] = 1.0
                A[nfit:, 1] = 1.0
                A[:nfit, 2] = -(np_c - np_mean)
                A[nfit:, 2] = nz_c - nz_mean

                b = np.concatenate([ez, ep])
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                rec = {'agc_camera_id': int(cid),
                       'dalt': x[0], 'daz': x[1],
                       'rotation': np.rad2deg(x[2]) * 3600.0}

                nz_all = dz_nom[sel]
                np_all = dp_nom[sel]
                cam_model_dz[sel] = x[0] + x[2] * (-(np_all - np_mean))
                cam_model_dp[sel] = x[1] + x[2] * (nz_all - nz_mean)
            else:
                mean_dz = np.mean(ez)
                mean_dp = np.mean(ep)
                rec = {'agc_camera_id': int(cid),
                       'dalt': mean_dz, 'daz': mean_dp,
                       'rotation': 0.0}
                cam_model_dz[sel] = mean_dz
                cam_model_dp[sel] = mean_dp

            cam_records.append(rec)

    out_dz = exp_model_dz + cam_model_dz
    out_dp = exp_model_dp + cam_model_dp

    _col_units = {'dalt': 'mm', 'daz': 'mm', 'rotation': 'arcsec', 'scale': ''}

    exp_df = pd.DataFrame(exp_records).set_index('agc_exposure_id')
    exp_df.units = {c: _col_units[c] for c in exp_df.columns if c in _col_units}
    exp_df._metadata.append('units')

    params = {'exposure': exp_df}
    if fitAgcOffsets:
        cam_df = pd.DataFrame(cam_records).set_index('agc_camera_id')
        cam_df.units = {c: _col_units[c] for c in cam_df.columns if c in _col_units}
        cam_df._metadata.append('units')
        params['camera'] = cam_df

    return out_dz, out_dp, params


def fit_global_model_pfimm(xpfi_nominal, ypfi_nominal, xpfi_centroid, ypfi_centroid,
                           inr, agc_exposure_id, agc_camera_id=None, valid=None,
                           fitPfiRotation=True, fitPfiScale=False,
                           fitAgcOffsets=False, fitAgcRotation=False):
    """Fit a global model in zenith/perp frame but return the model in PFI mm.

    The per-exposure and per-camera terms are fit in the zenith-aligned frame
    (where gravity flexure is constant), but the returned model vectors are
    rotated back to PFI mm coordinates for direct comparison with the input
    positions.

    Parameters
    ----------
    xpfi_nominal : array or Series
        Nominal x position in PFI mm (ag_pixel_to_pfimm convention).
    ypfi_nominal : array or Series
        Nominal y position in PFI mm (ag_pixel_to_pfimm convention).
    xpfi_centroid : array or Series
        Centroid x position in PFI mm.
    ypfi_centroid : array or Series
        Centroid y position in PFI mm.
    inr : float or array
        Instrument rotator angle in degrees.  If array, one per star (the
        per-exposure value is used for the rotation to/from zenith frame).
    agc_exposure_id : array or Series
        Exposure identifier.
    agc_camera_id : array or Series, optional
        Camera identifier [0, 5].  Required if fitAgcOffsets is True.
    valid : array of bool, optional
        Mask of stars to use in the fit (e.g. agc_match_flags == 1).
    fitPfiRotation : bool
        If True, fit a global rotation about the boresight per exposure.
    fitPfiScale : bool
        If True, fit a global scale change per exposure.
    fitAgcOffsets : bool
        If True, fit a constant (dz, dp) offset per camera across all exposures.
    fitAgcRotation : bool
        If True, also fit a constant rotation per camera.

    Returns
    -------
    model_x : ndarray
        Model displacement in PFI x (mm), ag_pixel_to_pfimm convention.
    model_y : ndarray
        Model displacement in PFI y (mm), ag_pixel_to_pfimm convention.
    params : dict
        'exposure': Styler (DataFrame) indexed by agc_exposure_id with columns
            dalt (mm), daz (mm), [rotation (arcsec)], [scale].
        'camera': Styler (DataFrame) indexed by agc_camera_id with columns
            dalt (mm), daz (mm), [rotation (arcsec)].
            Only present if fitAgcOffsets is True.
    """
    xpfi_nom = np.asarray(xpfi_nominal, dtype=float)
    ypfi_nom = np.asarray(ypfi_nominal, dtype=float)
    xpfi_cen = np.asarray(xpfi_centroid, dtype=float)
    ypfi_cen = np.asarray(ypfi_centroid, dtype=float)
    inr_arr = np.broadcast_to(np.asarray(inr, dtype=float), xpfi_nom.shape)
    exp = np.asarray(agc_exposure_id)

    N = len(xpfi_nom)

    # Convert to zenith/perp per-exposure (use one InR per exposure)
    dz_nom = np.empty(N)
    dp_nom = np.empty(N)
    dz_cen = np.empty(N)
    dp_cen = np.empty(N)
    inr_per_star = np.empty(N)

    for eid in np.unique(exp):
        sel = exp == eid
        inr_exp = np.median(inr_arr[sel])
        inr_per_star[sel] = inr_exp
        dz_nom[sel], dp_nom[sel] = ag_pfimm_to_zenith_offset(
            xpfi_nom[sel], ypfi_nom[sel], inr_exp)
        dz_cen[sel], dp_cen[sel] = ag_pfimm_to_zenith_offset(
            xpfi_cen[sel], ypfi_cen[sel], inr_exp)

    # Fit in zenith/perp frame
    model_dz, model_dp, params = fit_global_model(
        dz_nom, dp_nom, dz_cen, dp_cen,
        agc_exposure_id, agc_camera_id=agc_camera_id, valid=valid,
        fitPfiRotation=fitPfiRotation, fitPfiScale=fitPfiScale,
        fitAgcOffsets=fitAgcOffsets,
        fitAgcRotation=fitAgcRotation)

    # Rotate model back to PFI: dp = xfp, dz = yfp
    pfs = PFS()
    model_x, model_y = pfs.fp2pfi(model_dp, model_dz, inr_per_star)

    return model_x, model_y, params
