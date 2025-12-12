import numpy as np


def generate_awg_piezo_signals(
    x_um, y_um, z_um,
    scan,
    xy_scan_size_um, xy_scan_step_um,
    z_scan_size_um, z_scan_step_um,
    mv_per_um, awg_max_output_mv, awg_max_integer_value_ch0, signal_length,
    do_group_into_chunks=False
):
    """
    Generate analog signals for scan or no-scan displacements.

    Parameters:
    - x_um, y_um, z_um: floats
        Center spatial coordinates in micrometers.
    - scan: bool
        If True, perform snake scan generation; otherwise single position.
    - xy_scan_size_um, xy_scan_step_um: floats
        Scan size and step for XY directions in micrometers.
    - z_scan_size_um, z_scan_step_um: floats
        Scan size and step for Z direction in micrometers.
    - mv_per_um: float
        Voltage conversion factor (mV per micrometer).
    - awg_max_output_mv: float
        Maximum output voltage of the AWG in mV.
    - awg_max_integer_value_ch0: int
        Bit depth exponent for channel 0 (e.g. 14 for 14-bit).
    - signal_length: int
        Number of time samples per scan position.

    Returns:
    - analog_signal_ch0_off: ndarray (Nscan, signal_length), dtype=int16
        Channel 0 signals with MSB replaced by 0.
    - analog_signal_ch0_on: ndarray (Nscan, signal_length), dtype=int16
        Channel 0 signals with MSB replaced by 1.
    - analog_signal_ch1: ndarray (Nscan, signal_length), dtype=int16
        Channel 1 signals (unchanged).
    - analog_signal_ch2: ndarray (Nscan, signal_length), dtype=int16
        Channel 2 signals (unchanged).

    Notes:
    - Nscan = number of scan positions (1 if no scan).
    - Output arrays are 2D: rows correspond to scan positions, columns to time samples.
    - Channels correspond to spatial axes:
        * Channel 0: X coordinate
        * Channel 1: Y coordinate
        * Channel 2: Z coordinate
    """
    if not scan:
        # Single position case
        # displacements shape: (3,) --> [x_um, y_um, z_um]
        displacements = np.array([x_um, y_um, z_um])

        # Convert displacement (um) to voltage (mV), shape (3,)
        voltages_mv = mv_per_um * displacements

        # Normalize voltages to amplitudes (unitless), shape (3,)
        amplitudes = voltages_mv / awg_max_output_mv

        # Repeat amplitudes for all time samples, shape (signal_length, 3)
        amplitudes_signals = np.tile(amplitudes, (signal_length, 1))

        # Reshape to 3D: (Nscan=1, signal_length, channels=3)
        # Transpose and add axis to get shape (1, signal_length, 3)
        amplitudes_signals = amplitudes_signals[np.newaxis, :, :]

    else:
        # Scan case

        # Generate snake scan positions for X, Y, Z as 1D arrays of length Nscan
        X_flat, Y_flat, Z_flat = generate_snake_scan_positions(
            x_um, y_um, z_um,
            xy_scan_size_um, xy_scan_step_um,
            z_scan_size_um, z_scan_step_um
        )

        # Stack positions into shape (Nscan, 3)
        # Columns correspond to X, Y, Z coordinates respectively
        displacements = np.column_stack((X_flat, Y_flat, Z_flat))

        # Convert displacements to voltages (mV), shape (Nscan, 3)
        voltages_mv = mv_per_um * displacements

        # Normalize to amplitudes, shape (Nscan, 3)
        amplitudes = voltages_mv / awg_max_output_mv

        # Repeat amplitudes for signal_length time samples per scan position
        # Resulting shape: (Nscan, signal_length, 3)
        amplitudes_signals = np.repeat(amplitudes[:, np.newaxis, :], signal_length, axis=1)

    # Define scale factors for each channel (X, Y, Z) in order: channel 0, 1, 2
    scale_factors = np.array([2**(awg_max_integer_value_ch0-1) - 1, 2**15 - 1, 2**15 - 1])

    # Scale amplitudes by scale factors (broadcast over scan positions and time)
    # Result shape: (Nscan, signal_length, 3)
    analog_signals = amplitudes_signals * scale_factors[np.newaxis, np.newaxis, :]

    # Convert to int16 for downstream bitwise operations
    analog_signals = analog_signals.astype(np.int16)

    # Extract channel 0 signals (X axis), shape (Nscan, signal_length)
    ch0_signals = analog_signals[:, :, 0]

    # Replace MSB in channel 0 signals for off (0) and on (1)
    # np.apply_along_axis applies replace_msb_bit along time axis for each scan position
    analog_signal_ch0_off = np.apply_along_axis(replace_msb_bit, 1, ch0_signals, 0)
    analog_signal_ch0_on = np.apply_along_axis(replace_msb_bit, 1, ch0_signals, 1)

    # Channels 1 (Y axis) and 2 (Z axis) remain unchanged
    analog_signal_ch1 = analog_signals[:, :, 1]
    analog_signal_ch2 = analog_signals[:, :, 2]

    # if scan:
    #     # Number of times to prepend the initial signal to allow piezo to travel there
    #     num_initial_repeats = 50
    #
    #     # Prepare initial blocks
    #     initial_ch0_off = np.repeat(analog_signal_ch0_off[0:1], num_initial_repeats, axis=0)
    #     initial_ch0_on  = np.repeat(analog_signal_ch0_off[0:1], num_initial_repeats, axis=0)  # still OFF for initial
    #     initial_ch1     = np.repeat(analog_signal_ch1[0:1],     num_initial_repeats, axis=0)
    #     initial_ch2     = np.repeat(analog_signal_ch2[0:1],     num_initial_repeats, axis=0)
    #
    #     # Concatenate with rest of scan
    #     full_ch0_off = np.concatenate((initial_ch0_off, analog_signal_ch0_off), axis=0)
    #     full_ch0_on  = np.concatenate((initial_ch0_on,  analog_signal_ch0_on),  axis=0)
    #     full_ch1     = np.concatenate((initial_ch1,     analog_signal_ch1),     axis=0)
    #     full_ch2     = np.concatenate((initial_ch2,     analog_signal_ch2),     axis=0)
    #
    # else:
    full_ch0_off = analog_signal_ch0_off
    full_ch0_on = analog_signal_ch0_on
    full_ch1 = analog_signal_ch1
    full_ch2 = analog_signal_ch2

    if do_group_into_chunks:
        pixels_in_a_row = int(np.sqrt(full_ch0_on.shape[0]))
        return (np.ascontiguousarray(group_into_chunks(full_ch0_on, pixels_in_a_row)),
                np.ascontiguousarray(group_into_chunks(full_ch0_off, pixels_in_a_row)),
                np.ascontiguousarray(group_into_chunks(full_ch1, pixels_in_a_row)),
                np.ascontiguousarray(group_into_chunks(full_ch2, pixels_in_a_row)))
    else:
        return (np.ascontiguousarray(full_ch0_on[np.newaxis, :, :]),
                np.ascontiguousarray(full_ch0_off[np.newaxis, :, :]),
                np.ascontiguousarray(full_ch1[np.newaxis, :, :]),
                np.ascontiguousarray(full_ch2[np.newaxis, :, :]))


def generate_snake_scan_positions(
    X_position_um, Y_position_um, Z_position_um,
    XY_scan_size_um, XY_scan_step_um,
    Z_scan_size_um, Z_scan_step_um
):
    # Compute number of scan steps
    XY_n_steps = int(np.floor(XY_scan_size_um / XY_scan_step_um))
    Z_n_steps = int(np.floor(Z_scan_size_um / Z_scan_step_um))

    # Validate
    if XY_n_steps <= 0:
        raise ValueError(f"XY scan range too small: size={XY_scan_size_um}, step={XY_scan_step_um}")
    if Z_n_steps < 0:
        raise ValueError(f"Z scan range too small: size={Z_scan_size_um}, step={Z_scan_step_um}")

    # Generate scan positions
    scan_positions_x_um = np.linspace(
        X_position_um - XY_scan_size_um / 2,
        X_position_um + XY_scan_size_um / 2,
        XY_n_steps
    )
    scan_positions_y_um = np.linspace(
        Y_position_um - XY_scan_size_um / 2,
        Y_position_um + XY_scan_size_um / 2,
        XY_n_steps
    )
    scan_positions_z_um = (
        np.array([Z_position_um]) if Z_n_steps <= 1 else np.linspace(
            Z_position_um - Z_scan_size_um / 2,
            Z_position_um + Z_scan_size_um / 2,
            Z_n_steps
        )
    )

    # Create meshgrid
    Z_mesh, Y_mesh, X_mesh = np.meshgrid(
        scan_positions_z_um, scan_positions_y_um, scan_positions_x_um, indexing='ij'
    )

    # Flatten (X fastest)
    X_flat = X_mesh.ravel()
    Y_flat = Y_mesh.ravel()
    Z_flat = Z_mesh.ravel()

    return X_flat, Y_flat, Z_flat


def replace_msb_bit(int16_array, new_msb_bit):
    """
    Replace the most significant bit (bit 15) in each int16 element with msb_bit (0 or 1).

    Parameters:
        int16_array (np.ndarray): 1D array of int16 values.
        new_msb_bit (int): Bit value to set (0 or 1).

    Returns:
        np.ndarray: New int16 array with modified MSBs.
    """
    # Check that msb_bit is valid
    if new_msb_bit not in (0, 1):
        raise ValueError("msb_bit must be 0 or 1")

    # Convert input to numpy array with dtype int16 (if not already)
    int16_array = np.asarray(int16_array, dtype=np.int16)

    # View the same data as unsigned 16-bit integers so we can manipulate bits safely
    bits = int16_array.view(np.uint16)

    # Mask to clear the MSB (bit 15)
    clear_msb_mask = 0x7FFF  # binary 0111 1111 1111 1111

    # Clear the MSB bit in all values
    bits_cleared = bits & clear_msb_mask

    # If msb_bit is 1, set the MSB; otherwise, leave it cleared
    if new_msb_bit == 1:
        bits_modified = bits_cleared | 0x8000  # binary 1000 0000 0000 0000
    else:
        bits_modified = bits_cleared

    # Convert back to signed int16 before returning
    return bits_modified.view(np.int16)


def group_into_chunks(arr, chunk_length=2025):
    """
    Groups a 2D array (Nscan, signal_length) into 3D chunks of shape
    (n_chunks, chunk_length, signal_length), padding with zeros if necessary.

    Parameters:
    - arr: ndarray (Nscan, signal_length)
        The input signal array to be chunked.
    - chunk_length: int
        Number of scan positions per chunk.

    Returns:
    - grouped: ndarray (n_chunks, chunk_length, signal_length)
        The chunked and padded signal.
    """
    n_total = arr.shape[0]
    n_chunks = int(np.ceil(n_total / chunk_length))
    pad_size = n_chunks * chunk_length - n_total

    # Pad with zeros if needed to complete the last chunk
    if pad_size > 0:
        padding = np.zeros((pad_size, arr.shape[1]), dtype=arr.dtype)
        arr = np.vstack([arr, padding])

    # Reshape into chunks
    grouped = arr.reshape(n_chunks, chunk_length, arr.shape[1])
    return grouped
