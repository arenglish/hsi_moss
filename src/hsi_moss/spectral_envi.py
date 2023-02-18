#
# University of Eastern Finland,
# Color Science Laboratory course 2022
#
# Python codes for Homework #1.
#
# (c) Pauli Fält, Sep 27 2022
#

def read_envi(headerFileName, rawFileExt='.raw'):
    """
    dataCube, wavelengths = read_envi(headerFileName, rawFileExt='.raw')

    Reads ENVI data files. Returns 3-D spectral data cube and wavelengths.

    INPUTS:     'headerFileName' = Name of the ENVI header file 
                                   (with or without '.hdr' extention)

                'rawFileExt' = (optional) File extension of the raw data file 
                                          (default: '.raw').

    OUTPUTS:    'dataCube' = 3-D data cube containing the 
                             spectral bands along the 3rd axis.

                'wavelengths' = Wavelengths of the spectral bands.
    """

    import numpy as np

    if headerFileName[-4:] != '.hdr':
        # if header file name doesn't end in '.hdr', add it:
        headerFileName += '.hdr'

    # File name without extension:
    fileName = headerFileName[:-4]

    # Read header file:
    f = open(headerFileName, 'r')
    header = f.readlines()
    f.close()

    for line in header:
        line = line.lower().strip()
        if 'samples' in line:
            samples = int(line.split('samples = ')[1])
        if 'bands' in line and 'default' not in line:
            bands = int(line.split('bands = ')[1])
        if 'lines' in line:
            lines = int(line.split('lines = ')[1])
        if 'interleave' in line:
            interleave = line.split('interleave = ')[1]
        if 'byte order' in line:
            byteorder = int(line.split('byte order = ')[1])

    wavelengths = ''
    isWavelength = False

    for line in header:
        line = line.lower().strip()
        if 'wavelength =' in line:
            isWavelength = True
        if isWavelength:
            wavelengths += line

    f1 = wavelengths.find('{')
    f2 = wavelengths.find('}')
    wavelengths = wavelengths[f1+1:f2]
    wavelengths = [float(n) for n in wavelengths.split(',')]
    wavelengths = np.array(wavelengths)

    if byteorder == 0:  # Specim
        dataType = np.dtype('<u2')
    if byteorder == 1:  # Senop
        dataType = np.dtype('>u2')

    # Read raw data file:
    f = open(fileName + rawFileExt, 'r')
    raw_data = np.fromfile(f, dataType)
    f.close()

    # Reshape raw data into a 3-D data cube:
    if interleave == 'bil':
        dataCube = raw_data.reshape((lines, bands, samples))
        dataCube = dataCube.swapaxes(1, 2)
    elif interleave == 'bip':
        dataCube = raw_data.reshape((lines, samples, bands))
    elif interleave == 'bsq':
        dataCube = raw_data.reshape((bands, lines, samples))
        dataCube = dataCube.swapaxes(0, 1)
        dataCube = dataCube.swapaxes(1, 2)

    dataCube = dataCube.astype('float')

    return dataCube, wavelengths