import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# from DataSetGeneration import __convert_to_feature, __convert_to_tf_example


# ToDo: Sinewave long, sinewave short, superposition noise over Amplitude, superposition Gaussian over thi value with noisy variance

def generative_process_mean(x, f_long=365.25*24*60, f_short=24*60): #times are all in minutes
    arg_1 = np.pi*(1/f_long)*x
    arg_2 = np.pi * (1 / f_short) * x

    y = (np.sin(arg_1) - 0.5) + np.sin(arg_2) #0.5 so days are not infinitely long

    # crop everything below 0
    y = max(y, 0)
    return y

def __test_generative_process_mean():
    y = []
    for x in range(int(1e5)):
        y.append(generative_process_mean(x))
    plt.plot(y)
    plt.ylabel('amplitude')
    plt.xlabel('seconds')
    plt.show()
    print('visual test pls')

def generative_process_pdf(value_max, var=1, bins=30, bin_max=1, bin_min=0):
    p_x = 1/(np.sqrt(2*np.pi*var))
    p_bins = []
    bin_cutof = np.ceil(30 - (bin_max - value_max)*30/30)

    for bin in np.linspace(bin_min, bin_max, bins).tolist():
        if bin < bin_cutof:
            exp = -(bin)**2 / (2*var)
            p_bin = p_x * np.exp(exp)
            p_bins.append(p_bin)
        else:
            p_bins.append(0.0)

    return p_bins/sum(p_bins)

def downsample_pdf(pdf, downsample_factor):
    # check if length ok
    pdf = np.array(pdf)
    if pdf.shape[0]%downsample_factor != 0:
        print('not completely congruent downsampling desired, results might be wierd in the last timestep, consider cropping')
    pdf_downsampled = []
    for start_idx in range(0, pdf.shape[0], downsample_factor):
        pdf_slice = pdf[start_idx:(start_idx+downsample_factor),:]
        pdf_squish = np.sum(pdf_slice, axis=0)
        pdf_squish = pdf_squish/np.sum(pdf_squish)
        pdf_downsampled.append(pdf_squish.tolist())
    return np.array(pdf_downsampled)

def __test_generative_process_pdf():
    z = []
    y = []
    for x in range(int(60*24)):
        y.append(generative_process_mean(x))
    print('max valuee', np.amax(y))
    for y_value in y:
        print(y_value)
        z.append(generative_process_pdf(y_value))
    z = np.array(z)
    z = downsample_pdf(z, 60)
    X, Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,  linewidth = 0, antialiased = False)
    ax.view_init(90, 180)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    print('visual test pls')

