from pyVeReady.utils.image_utils import *
import scienceplots

def compute_means(measures):
    return [[np.mean(array) for array in measure] for measure in measures]

if __name__ == "__main__":
    data_path = ask_files_location('Select File')

    n_amplitudes = 26
    target_amplitudes = np.linspace(0, 1, n_amplitudes)

    data = tifffile.imread(data_path)
    list_of_measures = np.split(data, 10, axis = 0)
    list_of_measures_mean = compute_means(list_of_measures)

    plt.style.use(['science','no-latex', 'grid'])
    plt.figure(figsize=(6,6))
    colors = plt.cm.tab10.colors  # `tab10` has 10 distinct colors, adjust as needed
    normalize = True
    for i, measure_means in enumerate(list_of_measures_mean):
        measure_means = np.array(measure_means)
        if normalize:
            measure_means = measure_means - np.min(measure_means)
            measure_means = measure_means / np.max(measure_means)
        measure_means = np.sqrt(measure_means)
        plt.plot(target_amplitudes, measure_means, marker = 'o', color = colors[i], linestyle = '-', label = fr'$m_{{Size}}$ = {i + 1} Pix')

    if normalize:
        plt.title('Normalized Data')
        plt.axis('equal')
    else:
        plt.title('Raw Data')

    plt.xlabel("Target Amplitude")
    plt.ylabel(r"$\sqrt{I}$ (a.u.)")
    plt.legend(frameon = True, edgecolor = 'black')
    plt.show()