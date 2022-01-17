import numpy as np


# TODO add typing
# TODO add gitignore


def _get_mapping_function(mapping_dict):
    def mp(entry):
        return mapping_dict[entry] if entry in mapping_dict else entry

    return np.vectorize(mp)


def equalize_image(bw_image, considered_region=None):
    if considered_region is not None:
        intensities, counts = np.unique(bw_image[considered_region], return_counts=True)
        g_a, g_d = (
            np.mean(bw_image[considered_region]),
            np.std(bw_image[considered_region]),
        )
    else:
        intensities, counts = np.unique(bw_image, return_counts=True)
        g_a, g_d = np.mean(bw_image), np.std(bw_image)

    L = len(intensities)
    T_float = L * g_d / g_a
    T = int(round(T_float))

    N_left = counts[0:T].sum()
    N_right = counts[T:L].sum()

    T_l_cumsum = np.cumsum(counts[0:T])
    T_l = np.arange(0, T)[T_l_cumsum <= N_left / 2.0][-1] + 1

    T_u_cumsum = np.cumsum(counts[T:L])
    T_u = np.arange(T, L)[T_u_cumsum <= N_right / 2.0][-1] + 1

    Tm_L1 = np.median(counts[0:T_l])
    Tm_L2 = np.median(counts[T_l:T])
    Tm_U1 = np.median(counts[T:T_u])
    Tm_U2 = np.median(counts[T_u:L])

    Ta_L1 = counts[0:T_l].sum() / (T_l * 1.0)
    Ta_L2 = counts[T_l:T].sum() / (T_float - T_l) * 1.0
    Ta_U1 = counts[T:T_u].sum() / (T_u - T_float) * 1.0
    Ta_U2 = counts[T_u:L].sum() / (L - T_u) * 1.0

    T_L1 = Tm_L1 if Tm_L1 >= 0 else Ta_L1
    T_L2 = Tm_L2 if Tm_L2 >= 0 else Ta_L2
    T_U1 = Tm_U1 if Tm_U1 >= 0 else Ta_U1
    T_U2 = Tm_U2 if Tm_U2 >= 0 else Ta_U2

    hN_L1 = counts[0:T_l].copy()
    hN_L2 = counts[T_l:T].copy()
    hN_U1 = counts[T:T_u].copy()
    hN_U2 = counts[T_u:L].copy()

    hN_L1 = np.clip(hN_L1, 0, T_L1)
    hN_L2 = np.clip(hN_L2, 0, T_L2)
    hN_U1 = np.clip(hN_U1, 0, T_U1)
    hN_U2 = np.clip(hN_U2, 0, T_U2)

    P_L1 = hN_L1 / (1.0 * sum(hN_L1))
    P_L2 = hN_L2 / (1.0 * sum(hN_L2))
    P_U1 = hN_U1 / (1.0 * sum(hN_U1))
    P_U2 = hN_U2 / (1.0 * sum(hN_U2))

    C_L1 = np.cumsum(P_L1)
    C_L2 = np.cumsum(P_L2)
    C_U1 = np.cumsum(P_U1)
    C_U2 = np.cumsum(P_U2)

    f_1 = T_l * (C_L1 - 0.5 * P_L1)
    f_2 = (T_float - T_l) * (C_L2 - 0.5 * P_L2) + T_l
    f_3 = (T_u - T_float) * (C_U1 - 0.5 * P_U1) + T_float
    f_4 = (L - T_u) * (C_U2 - 0.5 * P_U2) + T_u

    new_intensities = intensities[
        np.clip(
            np.concatenate((f_1, f_2, f_3, f_4)).round().astype("int"),
            0,
            len(intensities) - 1,
        )
    ]

    new_intensities = np.uint8(
        (new_intensities - new_intensities.min())
        * 255.0
        / (new_intensities.max() - new_intensities.min())
    )

    mapping_dict = {key: value for key, value in zip(intensities, new_intensities)}

    return _get_mapping_function(mapping_dict)(bw_image.copy())
