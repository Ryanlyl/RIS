import numpy as np
import matplotlib.pyplot as plt


def main():
    # =========================================================
    # 1. Basic wave parameters
    # =========================================================
    c = 3e8
    f = 3.5e9
    wavelength = c / f
    k = 2 * np.pi / wavelength

    # =========================================================
    # 2. Tx / Rx positions
    # =========================================================
    # Transmitters
    Fd1 = np.array([0.0, 0.0, 2.5])
    Fd2 = np.array([0.0, 0.2, 2.0])

    # Receivers / focal points
    F1 = np.array([0.0, -0.4, 1.0])
    F2 = np.array([0.0,  0.4, 1.5])

    # =========================================================
    # 3. RIS element coordinates
    # =========================================================
    tx = np.arange((-0.0428 / 2 - 10 * 0.0428),
                   (+0.0428 / 2 + 10 * 0.0428) + 1e-12,
                   0.0428)

    ty = np.arange((-0.0431 / 2 - 10 * 0.0431),
                   (+0.0431 / 2 + 10 * 0.0431) + 1e-12,
                   0.0431)

    X, Y = np.meshgrid(tx, ty, indexing="ij")
    Z0 = np.zeros_like(X)

    # =========================================================
    # 4. Amplitude weights
    # =========================================================
    E1, E2= 1.0, 1.0

    # =========================================================
    # 5. Distance helper
    # =========================================================
    def distance_to_point(Xp, Yp, Zp, P):
        return np.sqrt((Xp - P[0])**2 + (Yp - P[1])**2 + (Zp - P[2])**2)

    # =========================================================
    # 6. Tx -> RIS distances
    # =========================================================
    d1 = distance_to_point(X, Y, Z0, Fd1)
    d2 = distance_to_point(X, Y, Z0, Fd2)

    # =========================================================
    # 7. RIS -> focal point distances
    # =========================================================
    foc1 = distance_to_point(X, Y, Z0, F1)
    foc2 = distance_to_point(X, Y, Z0, F2)

    # =========================================================
    # 8. Construct target phase WITHOUT OAM
    #    Original MATLAB:
    #    T_i = E_i * exp(j*k*d_i) * exp(j*l_i*phi) * exp(j*k*foc_i)
    #    Now remove exp(j*l_i*phi)
    # =========================================================
    T1 = E1 * np.exp(1j * k * d1) * np.exp(1j * k * foc1)
    T2 = E2 * np.exp(1j * k * d2) * np.exp(1j * k * foc2)

    # =========================================================
    # 9. Final RIS phase synthesis
    #    Keep same logic as MATLAB: only combine T1 and T2
    # =========================================================
    T = np.exp(1j * np.angle(T1 + T2))

    # RIS phase in [0, 2pi]
    phase_ris = np.angle(T) + np.pi

    # Equivalent reflected field on RIS
    T_amp = T
    U1 = T_amp * np.exp(-1j * k * d1)
    U2 = T_amp * np.exp(-1j * k * d2)

    # =========================================================
    # 10. Plot RIS amplitude / phase
    # =========================================================
    plt.figure(figsize=(6, 5))
    plt.imshow(np.abs(T_amp.T) / np.max(np.abs(T_amp)),
               extent=[tx.min(), tx.max(), ty.min(), ty.max()],
               origin="lower",
               aspect="auto")
    plt.colorbar()
    plt.title("RIS amplitude")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.figure(figsize=(6, 5))
    plt.imshow(phase_ris.T,
               extent=[tx.min(), tx.max(), ty.min(), ty.max()],
               origin="lower",
               aspect="auto")
    plt.colorbar()
    plt.title("RIS phase (0 to 2pi)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    # =========================================================
    # 11. Observation plane
    #     Observe field on x = 0 plane, varying y and z
    # =========================================================
    M = 100
    Tz = 100
    dR = 0.75
    Zmax = 2.0

    Ry = np.linspace(-dR, dR, M)
    Rz = np.linspace(0.0, Zmax, Tz)

    EM_beam1 = np.zeros((M, Tz), dtype=np.complex128)
    EM_beam2 = np.zeros((M, Tz), dtype=np.complex128)

    I1 = I2 = 1.0
    phi0 = 0.0

    # =========================================================
    # 12. Propagation from RIS to observation plane
    # =========================================================
    for iz, z_obs in enumerate(Rz):
        for iy, y_obs in enumerate(Ry):
            x_obs = 0.0

            dn = np.sqrt((x_obs - X)**2 + (y_obs - Y)**2 + (z_obs - Z0)**2)
            dn = np.maximum(dn, 1e-12)  # avoid divide-by-zero

            EM_beam1[iy, iz] = np.sum(
                I1 * np.exp(1j * phi0) * U1 * np.exp(-1j * k * dn) / np.sqrt(dn)
            )
            EM_beam2[iy, iz] = np.sum(
                I2 * np.exp(1j * phi0) * U2 * np.exp(-1j * k * dn) / np.sqrt(dn)
            )

    # Total field (same style as your MATLAB: beam1 + beam2)
    E_total = EM_beam1 + EM_beam2

    plt.figure(figsize=(7, 5))
    plt.imshow(np.abs(E_total),
               extent=[Rz.min(), Rz.max(), Ry.min(), Ry.max()],
               origin="lower",
               aspect="auto")
    plt.colorbar()
    plt.title("Field magnitude on y-z plane")
    plt.xlabel("z (m)")
    plt.ylabel("y (m)")

    # =========================================================
    # 13. 1-bit quantization for CST
    # =========================================================
    meta_cst = np.zeros_like(phase_ris, dtype=int)
    meta_cst[(phase_ris > np.pi) & (phase_ris <= 2 * np.pi)] = 1

    plt.figure(figsize=(6, 5))
    plt.imshow(meta_cst.T,
               extent=[tx.min(), tx.max(), ty.min(), ty.max()],
               origin="lower",
               aspect="auto")
    plt.colorbar()
    plt.title("1-bit RIS code")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    # MATLAB export swapped indices when writing
    np.savetxt("meta_cst_no_oam.txt", meta_cst.T, fmt="%d", delimiter="\t")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()