# Copyright (C) 2012-2015  ASTRON (Netherlands Institute for Radio Astronomy)
# P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
#
# This file is part of the LOFAR software suite.
# The LOFAR software suite is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The LOFAR software suite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.

"""Module hdf5_io offers various methods to read/write/modify hdf5 files containing lofar measurement data.
Such an h5 file is usually generated from Lofar Measurement Sets (MS/casacore format) using the ms2hdf5 conversion tool.

Since the data is stored in hdf (hierchical data format) and we use python, it makes sense that we use (nested) dicts as data holders.
The file contents is as follows:
- TODO

External developers using this api whill primarily use the read_hypercube.
If you would like to do your own clustering, then use write_clusters and read_clusters as well.

:Example:

    from lofar.qa.hdf5_io import *

    # read the data
    h5_path = '/my/path/to/myfile.h5'
    data = read_hypercube(h5_path, visibilities_in_dB=True, python_datetimes=False, read_flagging=False)

    # do your own processing, for example make clusters (see write_clusters for dict format)
    my_clusters = .... #results of your algorithm

    # write your clusters into the same h5 file
    # in this case they are stored under 'my_fancy_clustering_attempt_1', and a 'latest' symlink is made to these clustering results.
    # multiple clustering results can all be stored in the same file, each with a different algo_name.
    write_clusters(h5_path, clusters, algo_name='my_fancy_clustering_attempt_1')
"""

import os.path
from datetime import datetime, timedelta

import os
# prevent annoying h5py future/deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import h5py
import errno
import numpy as np
from time import sleep

import logging
logger = logging.getLogger(__name__)

np.set_printoptions(precision=1)

class SharedH5File():
    """
    Wrapper class aroung h5py.File to open an hdf5 file in read, write, or read/write mode safely,
    even when the file might be used simultanously by other processes.
    It waits for <timeout> seconds until the file becomes available.

    Example usage:

    with SharedH5File("foo.h5", 'r') as file:
        file["foo"] = "bar"

    """
    def __init__(self, path, mode='a', timeout=900):
        self._path = path
        self._mode = mode
        self._timeout = timeout
        self._file = None

    def open(self):
        start_timestamp = datetime.utcnow()
        while self._file is None:
            try:
                self._file = h5py.File(self._path, self._mode)
            except IOError as e:
                if not os.path.exists(self._path):
                    raise

                logger.warning("Cannot open file '%s' with mode '%s'. Trying again in 1 sec...",
                               self._path, self._mode)
                sleep(max(0, min(1, self._timeout)))
                if datetime.utcnow() - start_timestamp > timedelta(seconds=self._timeout):
                    logger.error("Cannot open file '%s' with mode '%s', even after trying for %s seconds",
                                   self._path, self._mode, self._timeout)
                    raise

        return self._file

    def close(self):
        self._file.close()
        self._file = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

def write_hypercube(path, saps, parset=None, sas_id=None, wsrta_id=None, do_compress=True, **kwargs):
    """
    write a hypercube of visibility/flagging data for all saps of an observation/pipeline.

    :param str path: full path of the resulting h5 file. By convention we advise to use <observation_id>.MS_exctract.h5
                     where observation_id is L<sas_id> for lofar and WSRTA<wsrta_id> for apertif
    :param dict saps: each key is the id of a sap, and holds per sap a dict with the following key/value pairs:

                          baselines: [string], list of stationpairs (tuples) (these are the ticks on the baseline axis of the visibilities)

                          timestamps: [np.double], list of Modified Julian Date (these are the ticks on the time axis of the visibilities)

                          central_frequencies: [np.double], list of central frequencies of the subbands (these are the ticks on the frequency axis of the visibilities)

                          subbands: [np.int], list of subbands numbers (each subband has a corresponding central_frequency)

                          polarizations: [string], list of polarization, one up to four, any of 'XX', 'XY', 'YX', 'YY'

                          visibilities: numpy.array, the 4D array of visibilities. In the file these are reduced from doubles to chars by taking the 10.log10 and normalizing the result to fit in the [-128..127] range.

                          flagging: numpy.array, the 4D array of flagging booleans.
    :param parameterset parset: the optional paramaterset with all the settings which were used for this observation/pipeline
    :param int sas_id: the optional observation/pipeline sas_id (the main id to track lofar observations/pipelines)
    :param int wsrta_id: the optional observation wsrta_id (the main id to track wsrt apertif observations)
    :param bool do_compress: compress the visibilities and flagging data (with lzf compression, slower but smaller output size)
    :param dict kwargs: optional extra arguments
    :return None
    """
    logger.info('writing hypercube to file: %s', path)

    save_dir = os.path.dirname(path)
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(os.getcwd(), save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with SharedH5File(path, "w") as file:
        version = '1.4'
        # 1.1 -> 1.2 change is not backwards compatible by design.
        # 1.2 -> 1.3 change is almost backwards compatible, it just needs a dB/linear correction. see convert_12_to_13
        # 1.3 -> 1.4 storing scale factors per baseline per subband per pol, see convert_13_to_14
        ds = file.create_dataset('version', (1,), h5py.special_dtype(vlen=str), version)
        ds.attrs['description'] = 'version of this hdf5 MS extract file'

        measurement_group = file.create_group('measurement')
        measurement_group.attrs['description'] = 'all data (visibilities, flagging, parset, ...) for this measurement (observation/pipeline)'

        if parset is not None:
            ds = file.create_dataset('measurement/parset', (1,), h5py.special_dtype(vlen=str),
                                     [str(parset).encode('utf-8')],
                                     compression="lzf")
            ds.attrs['description'] = 'the parset of this observation/pipeline with all settings how this data was created'

        if sas_id is not None:
            ds = file.create_dataset('measurement/sas_id', data=[sas_id])
            ds.attrs['description'] = 'lofar observation/pipeline sas id'

        if wsrta_id is not None:
            ds = file.create_dataset('measurement/wsrta_id', data=[wsrta_id])
            ds.attrs['description'] = 'apertif observation wsrta id'

        saps_group = file.create_group('measurement/saps')
        saps_group.attrs['description'] = 'the data (visibilities, flagging, ...) is stored per sub-array-pointing (sap)'

        for sap_nr in sorted(saps.keys()):
            sap_dict = saps[sap_nr]
            baselines = sap_dict['baselines']
            timestamps = sap_dict['timestamps']
            central_frequencies = sap_dict['central_frequencies']
            subbands = sap_dict['subbands']
            polarizations = sap_dict['polarizations']
            visibilities = sap_dict['visibilities']
            flagging = sap_dict['flagging']
            antenna_locations = sap_dict.get('antenna_locations')

            sap_group = file.create_group('measurement/saps/%d' % sap_nr)
            ds = sap_group.create_dataset('polarizations', (len(polarizations),), h5py.special_dtype(vlen=str),
                                          [p.encode('ascii') for p in polarizations])
            ds.attrs['description'] = 'polarizations of the visibilities'

            ds = sap_group.create_dataset('baselines', (len(baselines),2), h5py.special_dtype(vlen=str),
                                    [[bl[0].encode('ascii'), bl[1].encode('ascii')] for bl in baselines])
            ds.attrs['description'] = 'pairs of baselines between stations'

            if any(isinstance(t, datetime) for t in timestamps):
                # try to import lofar.common.datetimeutils here and not at the top of the file
                # to make this hdf5_io module as loosly coupled to other lofar code as possible
                # do raise the possible ImportError, because we cannot proceed without converted datetimes.
                from lofar.common.datetimeutils import to_modified_julian_date_in_seconds
                timestamps = [to_modified_julian_date_in_seconds(t) if isinstance(t, datetime) else t for t in timestamps]

            ds = sap_group.create_dataset('timestamps', data=timestamps)
            ds.attrs['units'] = 'modified julian date, (fractional) seconds since epoch 1858-11-17 00:00:00'

            ds = sap_group.create_dataset('central_frequencies', data=central_frequencies)
            ds.attrs['units'] = 'Hz'

            ds = sap_group.create_dataset('subbands', data=subbands)
            ds.attrs['description'] = 'subband number'

            if antenna_locations:
                location_group = sap_group.create_group('antenna_locations')
                location_group.attrs['description'] = 'the antenna locations in XYZ, PQR, WGS84 coordinates (units: meters and/or radians)'

                for ref_frame in ['XYZ', 'PQR', 'WGS84']:
                    location_sub_group = location_group.create_group(ref_frame)
                    location_sub_group.attrs['description'] = 'the antenna locations in %s coordinates (units: meters and/or radians)' % (ref_frame,)

                    for antenna, location in antenna_locations[ref_frame].items():
                        location_sub_group.create_dataset(antenna, data=location)

            logger.debug('''flagging NaN's and zero's in visibilities for file %s''', path)
            zero_or_nan = np.absolute(visibilities) == 0.0
            zero_or_nan[np.isnan(visibilities)] = True
            flagging[zero_or_nan] = True

            #we'll scale the 10log10(visibilities) so the complex-float can be mapped onto 2*int8
            logger.debug('normalizing visibilities for file %s', path)
            #remove any NaN and/or 0 values in the visibilities? log(0) or log(nan) crashes,
            # so determine smallest non-zero abs value, and fill that in for the flagged visibilities
            try:
                abs_non_zero_or_nan_visibilities = np.abs(visibilities)[zero_or_nan == False]
                min_non_zero_or_nan_abs_value = max(1e-9, np.min(abs_non_zero_or_nan_visibilities))
                del abs_non_zero_or_nan_visibilities
            except ValueError:
                min_non_zero_or_nan_abs_value = 1e-12

            # overwrite all visibilities values where flagging (or 0's or NaN's) occur with the min_non_flagged_value
            # that enables us to take the log, and have good dynamic range when scaling to -128...127
            visibilities[zero_or_nan] = min_non_zero_or_nan_abs_value
            del zero_or_nan

            # reduce dynamic range (so we fit more data in the available bits)
            visibility_amplitudes = np.abs(visibilities)
            visibility_amplitudes_dB = 10.0*np.log10(visibility_amplitudes)
            visibility_phases = np.exp(1j*np.angle(visibilities))
            visibilities_dB = visibility_amplitudes_dB * visibility_phases

            #compute scale factors per subband, per polarization
            scale_factors = np.empty(shape=(len(baselines),len(subbands),len(polarizations)), dtype=np.float32)

            # compute scale factor per baseline/subband/pol to map the visibilities_dB from complex64 to 2xint8
            for bl_idx in range(len(baselines)):
                for pol_idx in range(len(polarizations)):
                    for sb_idx in range(len(subbands)):
                        #use 99.5 percentile instead if max to get rid of spikes
                        max_abs_vis_sb = max(1.0, np.percentile(visibility_amplitudes_dB[bl_idx,:,sb_idx,pol_idx], 99.5))
                        scale_factor = 127.0 / max_abs_vis_sb
                        scale_factors[bl_idx, sb_idx, pol_idx] = 1.0/scale_factor

            # store the scale_factors in the file
            scale_factor_ds = sap_group.create_dataset('visibility_scale_factors', data=scale_factors)
            scale_factor_ds.attrs['description'] = 'scale factors per baseline per subband per polatization to un-normalize the stored visibilities'
            scale_factor_ds.attrs['description'] = 'multiply real and imag parts of the visibilities with this factor per baseline per subband per polatization to un-normalize them and get the 10log10 values of the real and imag parts of the visibilities'
            scale_factor_ds.attrs['units'] = '-'

            # create a array with one extra dimension, so we can split the complex value into two scaled int8's for real and imag part
            # looping in python is not the most cpu efficient way
            # but is saves us extra copies of the large visibilities array, which might not fit in memory?
            logger.debug('converting visibilities from complexfloat to 2xint8 for file %s', path)
            extended_shape = visibilities_dB.shape[:] + (2,)
            scaled_visibilities = np.empty(extended_shape, dtype=np.int8)

            for bl_idx in range(len(baselines)):
                for pol_idx in range(len(polarizations)):
                    for sb_idx in range(len(subbands)):
                        scale_factor = 1.0 / scale_factors[bl_idx, sb_idx, pol_idx]
                        scaled_visibilities[bl_idx,:,sb_idx,pol_idx,0] = scale_factor*visibilities_dB[bl_idx,:,sb_idx,pol_idx].real
                        scaled_visibilities[bl_idx,:,sb_idx,pol_idx,1] = scale_factor*visibilities_dB[bl_idx,:,sb_idx,pol_idx].imag

            logger.debug('reduced visibilities size from %s to %s bytes (factor %s)',
                         visibilities.nbytes, scaled_visibilities.nbytes, visibilities.nbytes/scaled_visibilities.nbytes)

            ds = sap_group.create_dataset('visibilities', data=scaled_visibilities,
                                          compression="lzf" if do_compress else None)
            ds.attrs['units'] = 'normalized dB within [-128..127]'
            ds.attrs['dim[0]'] = 'baselines'
            ds.attrs['dim[1]'] = 'timestamps'
            ds.attrs['dim[2]'] = 'central_frequencies & subbands'
            ds.attrs['dim[3]'] = 'polarizations'
            ds.attrs['dim[4]'] = 'real part of normalized within [-128..127] 10log10(visibilities)'
            ds.attrs['dim[5]'] = 'imag part of normalized within [-128..127] 10log10(visibilities)'

            ds = sap_group.create_dataset('flagging', data=flagging,
                                          compression="lzf" if do_compress else None)
            ds.attrs['units'] = 'bool (true=flagged)'
            ds.attrs['dim[0]'] = 'baselines'
            ds.attrs['dim[1]'] = 'timestamps'
            ds.attrs['dim[2]'] = 'central_frequencies & subbands'
            ds.attrs['dim[3]'] = 'polarizations'
            ds.attrs['dim[4]'] = 'flagging values'

    if parset is not None:
        fill_info_folder_from_parset(path)

    try:
        # try to import the lofar.common.util.humanreadablesize here and not at the top of the file
        # to make this hdf5_io module as loosly coupled to other lofar code as possible
        from lofar.common.util import humanreadablesize
        logger.info('finished writing %s hypercube to file: %s', humanreadablesize(os.path.getsize(path)), path)
    except ImportError:
        logger.info('finished writing hypercube to file: %s', path)


def read_sap_numbers(path):
    """
    read the sap numbers (keys) from the hypercube data from the hdf5 hypercube file given by path.
    :param str path: path to the hdf5 file you want to read
    :return list: list of sap numbers
    """
    logger.info('reading sap numbers from from file: %s', path)

    with SharedH5File(path, "r") as file:
        version_str = read_version(path)

        if version_str not in ['1.2', '1.3', '1.4']:
            raise ValueError('Cannot read version %s' % (version_str,))

        return sorted([int(sap_nr) for sap_nr in file['measurement/saps'].keys()])

def read_version(h5_path):
    with SharedH5File(h5_path, "r") as file:
        version = file['version'][0]
        if isinstance(version, bytes):
            return version.decode('utf-8')
        return version

def read_hypercube(path, visibilities_in_dB=True, python_datetimes=False, read_visibilities=True, read_flagging=True, saps_to_read=None):
    """
    read the hypercube data from the hdf5 hypercube file given by path.

    :param str path: path to the hdf5 file you want to read
    :param bool visibilities_in_dB: return the in dB scale, or linear scale.
    :param bool python_datetimes: return the timestamps as python datetime's when True (otherwise modified_julian_date/double)
    :param bool read_visibilities: do/don't read visibilities (can save read-time and memory usage)
    :param bool read_flagging: do/don't read flagging (can save read-time and memory usage)
    :param list saps_to_read: only read these given SAPs (can save read-time and memory usage)
    :return dict: same dict structure as in write_hypercube, parameter saps.
    seealso:: write_hypercube
    """
    logger.info('reading hypercube from file: %s', path)

    if read_version(path) == '1.2':
        convert_12_to_13(path)

    if read_version(path) == '1.3':
        convert_13_to_14(path)

    # reopen file read-only for safety reasons.
    with SharedH5File(path, "r") as file:
        if read_version(path) != '1.4':
            raise ValueError('Cannot read version %s' % (file['version'][0],))

        result = {}
        if 'measurement/parset' in file:
            parset = read_hypercube_parset(path)
            if parset:
                result['parset'] = parset

        if 'measurement/sas_id' in file:
            result['sas_id'] = file['measurement/sas_id'][0]

        if 'measurement/wsrta_id' in file:
            result['wsrta_id'] = file['measurement/wsrta_id'][0]

        result['saps'] = {}

        for sap_nr, sap_dict in file['measurement/saps'].items():
            sap_nr = int(sap_nr)
            if saps_to_read and sap_nr not in saps_to_read:
                continue

            sap_result = {}
            result['saps'][sap_nr] = sap_result

            polarizations = list(sap_dict['polarizations'])
            sap_result['polarizations'] = polarizations

            baselines = sap_dict['baselines'][:]
            baselines = [(bl[0], bl[1]) for bl in baselines]
            sap_result['baselines'] = baselines

            timestamps = sap_dict['timestamps'][:]
            if python_datetimes:
                try:
                    # try to import lofar.common.datetimeutils here and not at the top of the file
                    # to make this hdf5_io module as loosly coupled to other lofar code as possible
                    from lofar.common.datetimeutils import from_modified_julian_date_in_seconds
                    timestamps = [from_modified_julian_date_in_seconds(t) for t in timestamps]
                except ImportError as e:
                    logger.warning("Could not convert timestamps from modified julian date to python datetimes.")

            sap_result['timestamps'] = timestamps

            central_frequencies = sap_dict['central_frequencies'][:]
            sap_result['central_frequencies'] = central_frequencies

            subbands = sap_dict['subbands'][:]
            sap_result['subbands'] = subbands

            sap_result['antenna_locations'] = {}
            if 'antenna_locations' in sap_dict:
                location_group = sap_dict['antenna_locations']
                for ref_frame, location_sub_group in location_group.items():
                    sap_result['antenna_locations'][ref_frame] = {}
                    for antenna, location in location_sub_group.items():
                        sap_result['antenna_locations'][ref_frame][antenna] = tuple(location)

            if read_flagging:
                flagging = sap_dict['flagging'][:]
                sap_result['flagging'] = flagging

            if read_visibilities:
                # read the visibility_scale_factors and (scaled_)visibilities
                # denormalize them and convert back to complex
                scale_factors = sap_dict['visibility_scale_factors'][:]
                normalized_visibilities = sap_dict['visibilities'][:]

                logger.debug('denormalizing and converting real/imag to complex visibilities for file sap %s in %s', sap_nr, path)
                reduced_shape = normalized_visibilities.shape[:-1]
                visibilities = np.empty(reduced_shape, dtype=np.complex64)

                for bl_idx in range(len(baselines)):
                    for sb_idx in range(len(subbands)):
                        for pol_idx in range(len(polarizations)):
                            scale_factor = scale_factors[bl_idx, sb_idx, pol_idx]
                            visibilities[bl_idx,:,sb_idx,pol_idx].real = scale_factor*normalized_visibilities[bl_idx,:,sb_idx,pol_idx,0]
                            visibilities[bl_idx,:,sb_idx,pol_idx].imag = scale_factor*normalized_visibilities[bl_idx,:,sb_idx,pol_idx,1]

                if not visibilities_in_dB:
                    logger.debug('converting visibilities from dB to raw linear for file sap %s in %s', sap_nr, path)
                    visibilities = np.power(10, 0.1*np.abs(visibilities)) * np.exp(1j * np.angle(visibilities))

                #HACK: explicitely set non-XX-polarizations to 0 for apertif
                if 'measurement/wsrta_id' in file:
                    visibilities[:,:,:,1:] = 0

                if 'flagging' in sap_result:
                    #explicitely set flagged visibilities to 0
                    visibilities[sap_result['flagging']] = 0.0

                sap_result['visibilities'] = visibilities
                sap_result['visibilities_in_dB'] = visibilities_in_dB

            antennae = set([bl[0] for bl in sap_result['baselines']] + [bl[1] for bl in sap_result['baselines']])

            logger.info('sap: %s, #subbands: %s, #timestamps: %s, #baselines: %s, #antennae: %s, #polarizations: %s',
                        sap_nr,
                        len(sap_result['subbands']),
                        len(sap_result['timestamps']),
                        len(sap_result['baselines']),
                        len(antennae),
                        len(sap_result['polarizations']))

    logger.info('finished reading hypercube from file: %s', path)

    return result

def convert_12_to_13(h5_path):
    with SharedH5File(h5_path, "r+") as file:
        version_str = read_version(h5_path)

        if version_str != '1.2':
            raise ValueError('Cannot convert version %s to 1.3' % (version_str,))

        logger.info("converting %s from version %s to 1.3", h5_path, version_str)

        for sap_nr, sap_group in file['measurement/saps'].items():
            # read the scale_factors and visibilities in a v1.2 way,
            # including incorrect reverse log10 to undo the incorrect storage of phases
            scale_factors = sap_group['visibility_scale_factors'][:]
            normalized_visibilities = sap_group['visibilities'][:]
            subbands = sap_group['subbands']
            polarizations = sap_group['polarizations']

            # apply v1.2 reconstruction of visibilities
            visibilities = np.empty(normalized_visibilities.shape[:-1], dtype=np.complex64)
            for sb_nr, scale_factor in enumerate(scale_factors):
                visibilities[:, :, sb_nr, :].real = scale_factor * normalized_visibilities[:, :, sb_nr, :, 0]
                visibilities[:, :, sb_nr, :].imag = scale_factor * normalized_visibilities[:, :, sb_nr, :, 1]
            visibilities = np.power(10, 0.1 * visibilities)

            # now we have the original raw visibilities again (including some minor errors in amplitude and phase due to rounding/truncation.
            # let's store them in the correct v1.3 way.

            # reduce dynamic range (so we fit more data in the available bits)
            visibility_amplitudes = np.abs(visibilities)
            visibility_amplitudes_dB = 10.0*np.log10(visibility_amplitudes)
            visibility_phases = np.exp(1j*np.angle(visibilities))
            visibilities_dB = visibility_amplitudes_dB * visibility_phases

            #compute scale factors per subband, per polarization
            scale_factors = np.empty(shape=(len(subbands),len(polarizations)), dtype=np.float32)

            for pol_idx, polarization in enumerate(polarizations):
                #compute scale factor per subband to map the visibilities_dB per subband from complex64 to 2xint8
                for sb_nr in range(len(subbands)):
                    #use 99.9 percentile instead if max to get rid of spikes
                    max_abs_vis_sb = max(1.0, np.percentile(visibility_amplitudes_dB[:,:,sb_nr,pol_idx], 99.9))
                    scale_factor = 127.0 / max_abs_vis_sb
                    scale_factors[sb_nr, pol_idx] = 1.0/scale_factor

            # overwrite the scale_factors in the file
            del sap_group['visibility_scale_factors']
            scale_factor_ds = sap_group.create_dataset('visibility_scale_factors', data=scale_factors)
            scale_factor_ds.attrs['description'] = 'scale factors per subband per polatization to un-normalize the stored visibilities'
            scale_factor_ds.attrs['description'] = 'multiply real and imag parts of the visibilities with this factor per subband per polatization to un-normalize them and get the 10log10 values of the real and imag parts of the visibilities'
            scale_factor_ds.attrs['units'] = '-'

            # scale the visibilities in the v1.3 way
            extended_shape = visibilities_dB.shape[:] + (2,)
            scaled_visibilities = np.empty(extended_shape, dtype=np.int8)
            for sb_nr in range(len(subbands)):
                scale_factor = 1.0 / scale_factors[sb_nr]
                scaled_visibilities[:,:,sb_nr,:,0] = scale_factor*visibilities_dB[:,:,sb_nr,:].real
                scaled_visibilities[:,:,sb_nr,:,1] = scale_factor*visibilities_dB[:,:,sb_nr,:].imag

            # overwrite the scale_factors in the file
            sap_group['visibilities'][:] = scaled_visibilities

        # and finally update the version number
        file['version'][0] = '1.3'

        logger.info("converted %s from version %s to 1.3", h5_path, version_str)

def convert_13_to_14(h5_path):
    with SharedH5File(h5_path, "r+") as file:
        version_str = read_version(h5_path)

        if version_str != '1.3':
            raise ValueError('Cannot convert version %s to 1.4' % (version_str,))

        logger.info("converting %s from version %s to 1.4", h5_path, version_str)

        for sap_nr, sap_group in file['measurement/saps'].items():
            # read the scale_factors and visibilities in a v1.2 way,
            # including incorrect reverse log10 to undo the incorrect storage of phases
            scale_factors = sap_group['visibility_scale_factors'][:]
            baselines = sap_group['baselines']
            subbands = sap_group['subbands']
            polarizations = sap_group['polarizations']

            # apply v1.3 scale factors to new v1.4
            # in v1.3 scale factors were stored per subband per pol
            # in v1.4 scale factors are stored per baseline per subband per pol
            scale_factors_new = np.empty(shape=(len(baselines),len(subbands),len(polarizations)), dtype=np.float32)

            for pol_idx in range(len(polarizations)):
                for sb_idx in range(len(subbands)):
                    scale_factors_new[:,sb_idx,pol_idx] = scale_factors[sb_idx,pol_idx]

            # overwrite the scale_factors in the file
            del sap_group['visibility_scale_factors']
            scale_factor_ds = sap_group.create_dataset('visibility_scale_factors', data=scale_factors_new)
            scale_factor_ds.attrs['description'] = 'scale factors per baseline per subband per polatization to un-normalize the stored visibilities'
            scale_factor_ds.attrs['description'] = 'multiply real and imag parts of the visibilities with this factor per baseline per subband per polatization to un-normalize them and get the 10log10 values of the real and imag parts of the visibilities'
            scale_factor_ds.attrs['units'] = '-'

        # and finally update the version number
        file['version'][0] = '1.4'

        logger.info("converted %s from version %s to 1.4", h5_path, version_str)

def add_parset_to_hypercube(h5_path, otdbrpc):
    """
    helper method which tries to get the parset for the sas_id in the h5 file from otdb via the otdbrpc, and add it to the h5 file.

    :param str h5_path: path to the hdf5 file
    :param lofar.sas.otdb.otdbrpc.OTDBRPC otdbrpc: an instance of a OTDBPC client
    """
    try:
        with SharedH5File(h5_path, "r+") as file:
            if 'measurement/sas_id' in file:
                sas_id = file['measurement/sas_id'][0]

                logger.info('trying to get the parset for sas_id %s', sas_id)
                parset = otdbrpc.taskGetSpecification(otdb_id=sas_id)["specification"]

                if parset:
                    if 'measurement/parset' in file:
                        logger.info('removing previous parset from file')
                        del file['measurement/parset']

                    logger.info('adding parset for sas_id %s to %s hdf5 file', sas_id, os.path.basename(h5_path))
                    parset_str = '\n'.join(['%s=%s'%(k,parset[k]) for k in sorted(parset.keys())])
                    ds = file.create_dataset('measurement/parset', (1,), h5py.special_dtype(vlen=str), parset_str,
                                             compression="lzf")
                    ds.attrs['description'] = 'the parset of this observation/pipeline with all settings how this data was created'
                    logger.info('added parset for sas_id %s to %s hdf5 file', sas_id, os.path.basename(h5_path))

        fill_info_folder_from_parset(h5_path)
    except Exception as e:
        logger.error(e)


def read_hypercube_parset(h5_path, as_string=False):
    """
    read the measurement parset from the given hdf5 hypercube file

    :param str h5_path: path to the hdf5 file
    :param bool as_string: return the parset as string instead of as parameterset object if true
    :return parameterset/string: the parset (as string or as parameterset) if any, else None
    """
    logger.info('reading parset from %s hdf5 file', os.path.basename(h5_path))
    with SharedH5File(h5_path, "r") as file:
        if 'measurement/parset' in file:
            parset_str = file['measurement/parset'][0]
            if as_string:
                return '\n'.join(sorted(line.strip() for line in parset_str.split('\n')))

            # try to import the lofar.parameterset here and not at the top of the file
            # to make this hdf5_io module as loosly coupled to other lofar code as possible
            try:
                from lofar.parameterset import parameterset
                parset = parameterset.fromString(parset_str)
                return parset
            except ImportError as e:
                logger.warning("could not import parset: %s", e)

def get_observation_id_str(data):
    if 'sas_id' in data:
        return 'L%d' % data['sas_id']
    if 'wsrta_id' in data:
        return 'WSRTA%d' % data['wsrta_id']
    return 'unknown_id'

def get_default_h5_filename(data, timestamped_if_unknown=True):
    obs_id = get_observation_id_str(data)
    if 'unknown' in obs_id and timestamped_if_unknown:
        return datetime.utcnow().strftime('%Y%m%d%H%M%s') + '.MS_extract.h5'
    return obs_id + '.MS_extract.h5'

def combine_hypercubes(input_paths, output_dir, output_filename=None, do_compress=True):
    """
    combine list of hypercubes into one file, for example when you created many h5 file in parallel with one subband per file.
    :param [str] input_paths: paths of the hdf5 files you want to read and combine
    :param str output_dir: directory where to save the resulting combined h5 file
    :param str output_filename: optional output filename. if None, then <get_observation_id_str>.MS_extract.h5 is used
    :param bool do_compress: compress the visibilities and flagging data (with lzf compression, slower but smaller output size)
    """
    input_files = []
    output_path = None
    try:
        input_paths = sorted(input_paths)
        existing_paths = [p for p in input_paths if os.path.exists(p)]
        if not existing_paths:
            raise ValueError('No input h5 files with valid paths given: %s' % (', '.join(input_paths),))

        # convert any 1.2 to 1.3 file if needed
        for path in existing_paths:
            with SharedH5File(path, "r") as file:
                if read_version(path) == '1.2':
                    convert_12_to_13(path)

        # convert any 1.3 to 1.4 file if needed
        for path in existing_paths:
            with SharedH5File(path, "r") as file:
                if read_version(path) == '1.3':
                    convert_13_to_14(path)

        input_files = [SharedH5File(p, "r").open() for p in existing_paths]

        versions = set([file['version'][0] for file in input_files])

        if len(versions) != 1:
            raise ValueError('Cannot combine h5 files of multiple versions: %s' % (', '.join(versions),))

        version_str = list(versions)[0]

        if version_str != '1.4':
            raise ValueError('Cannot read version %s' % (version_str,))

        sas_ids = set([file['measurement/sas_id'][0] for file in input_files if 'measurement/sas_id' in file])
        if len(sas_ids) > 1:
            raise ValueError('Cannot combine h5 files of multiple observations with multiple sas_ids: %s' % (', '.join(sas_ids),))
        sas_id = list(sas_ids)[0] if sas_ids else None

        wsrta_ids = set([file['measurement/wsrta_id'][0] for file in input_files if 'measurement/wsrta_id' in file])
        if len(wsrta_ids) > 1:
            raise ValueError('Cannot combine h5 files of multiple observations with multiple wsrta_ids: %s' % (', '.join(wsrta_ids),))
        wsrta_id = list(wsrta_ids)[0] if wsrta_ids else None

        if output_filename is None:
            output_filename = get_default_h5_filename({'sas_id':sas_id} if sas_id else
                                                      {'wsrta_id': wsrta_id} if wsrta_id else None)

        output_path = os.path.join(output_dir, output_filename)
        logger.info('combine_hypercubes: combining %s h5 files into %s', len(input_paths), output_path)

        with SharedH5File(output_path, "w") as output_file:
            version = '1.4'
            ds = output_file.create_dataset('version', (1,), h5py.special_dtype(vlen=str), version)
            ds.attrs['description'] = 'version of this hdf5 MS extract file'

            measurement_group = output_file.create_group('measurement')
            measurement_group.attrs['description'] = 'all data (visibilities, flagging, parset, ...) for this measurement (observation/pipeline)'

            if sas_id is not None:
                ds = output_file.create_dataset('measurement/sas_id', data=[sas_id])
                ds.attrs['description'] = 'observation/pipeline sas id'

            #copy parset from the first input file containing one. assume parset is equal in all input files.
            try:
                input_file = next(f for f in input_files if 'measurement/parset' in f)
                h5py.h5o.copy(input_file.id, 'measurement/parset', output_file.id, 'measurement/parset')
            except StopIteration:
                pass #no input file with parset, so nothing to copy.

            #make saps group and description
            saps_group = output_file.create_group('measurement/saps')
            saps_group.attrs['description'] = 'the data (visibilities, flagging, ...) is stored per sub-array-pointing (sap)'

            #rest of the items are multi dimensional, and may have different dimensions across the input files (only along the subband axis)
            #gather item values of all files, per sap, then combine, then write in output_file
            value_dicts_per_sap = {}
            for input_file in input_files:
                logger.info('combine_hypercubes: parsing file %s', input_file.filename)

                for sap_nr, sap_dict in input_file['measurement/saps'].items():
                    sap_nr = int(sap_nr)
                    logger.info('combine_hypercubes:   parsing sap %d in file %s', sap_nr, input_file.filename)

                    #gather all items of one sap of one file in one dict
                    file_sap_value_dict = {}

                    for item in sap_dict.keys():
                        key = 'measurement/saps/%s/%s' % (sap_nr, item)
                        if item == 'antenna_locations':
                            file_sap_value_dict[key] = {}
                            location_group = sap_dict['antenna_locations']
                            for ref_frame, location_sub_group in location_group.items():
                                file_sap_value_dict[key][ref_frame] = {}
                                for antenna, location in location_sub_group.items():
                                    file_sap_value_dict[key][ref_frame][antenna] = location
                        else:
                            file_sap_value_dict[key] = input_file[key][:]

                    #now, all items of this sap in input_file have been gathered into file_sap_value_dict
                    #this sap of this input file may contain mutiple subbands
                    #split out file_value_dict per subband
                    if sap_nr not in value_dicts_per_sap:
                        #per sap we make lists of value_dicts (one value_dict per file)
                        #we'll sort and combine them later
                        value_dicts_per_sap[sap_nr] = []

                    num_subbands_in_sap_in_input_file = len(file_sap_value_dict['measurement/saps/%s/subbands' % (sap_nr,)])
                    logger.info('combine_hypercubes:   num_subbands=%d in sap %d in file %s', num_subbands_in_sap_in_input_file, sap_nr, input_file.filename)

                    for sb_cntr in range(num_subbands_in_sap_in_input_file):
                        value_dict = {}
                        for key,data in file_sap_value_dict.items():
                            if 'visibilities' in key:
                                value_dict[key] = data[:,:,sb_cntr,:,:]
                            elif 'flagging' in key:
                                value_dict[key] = data[:,:,sb_cntr,:]
                            elif any(item in key for item in ['baselines', 'polarizations', 'timestamps', 'antenna_locations']):
                                value_dict[key] = data
                            elif 'visibility_scale_factors' in key:
                                value_dict[key] = data[:, sb_cntr,:]
                            else:
                                value_dict[key] = data[sb_cntr]

                        #append the value_dict holding the items of a single subband to the subband list of this sap
                        value_dicts_per_sap[sap_nr].append(value_dict)

            logger.info('combine_hypercubes: sorting and combining all subbands and saps into one output file: %s', output_path)

            #all saps and all subbands have been parsed and put into value_dicts_per_sap
            #sort and combine them
            for sap_nr,sap_value_dicts in value_dicts_per_sap.items():
                num_subbands = len(sap_value_dicts)
                logger.info('combine_hypercubes:   sorting and combining %d subbands for sap %d', num_subbands, sap_nr)
                #sort the sap_value_dicts by subband
                sap_value_dicts = sorted(sap_value_dicts, key=lambda x: x['measurement/saps/%s/subbands' % (sap_nr,)])

                #combine all seperate subbands
                if sap_value_dicts:
                    combined_value_dict = {}
                    #setup numpy arrays based on shape and type of first value_dict, extend sb dimension to num_subbands
                    for key,data in sap_value_dicts[0].items():
                        if 'visibilities' in key or 'flagging' in key:
                            shape = list(data.shape)
                            shape.insert(2, num_subbands)
                            shape = tuple(shape)
                        elif 'visibility_scale_factors' in key:
                            # from (#bl,#pol) to (#bl,#sb,#pol)
                            shape = (data.shape[0], num_subbands, data.shape[1])
                        else:
                            shape = (num_subbands,)

                        if 'antenna_locations' not in key:
                            combined_value_dict[key] = np.empty(shape=shape, dtype=data.dtype)

                    #now loop over all value_dicts and copy data to it's subband slice in the just created empty numpy arrays
                    for sb_cntr, value_dict in enumerate(sap_value_dicts):
                        for key,data in value_dict.items():
                            if 'visibilities' in key:
                                combined_value_dict[key][:,:,sb_cntr,:,:] = data
                            elif 'flagging' in key:
                                combined_value_dict[key][:,:,sb_cntr,:] = data
                            elif any(item in key for item in ['baselines', 'polarizations', 'timestamps', 'antenna_locations']):
                                combined_value_dict[key] = data
                            elif 'visibility_scale_factors' in key:
                                combined_value_dict[key][:,sb_cntr,:] = data
                            else:
                                combined_value_dict[key][sb_cntr] = data

                    for key,data in combined_value_dict.items():
                        logger.info('combine_hypercubes:   storing %s in %s', key, output_filename)
                        ds_out = None
                        if 'visibilities' in key or 'flagging' in key:
                            ds_out = output_file.create_dataset(key, data=data,
                                                                compression="lzf" if do_compress else None)
                        elif 'antenna_locations' in key:
                            location_group = output_file.create_group(key)
                            location_group.attrs['description'] = 'the antenna locations in XYZ, PQR, WGS84 coordinates (units: meters and/or radians)'
                            for ref_frame, antenna_locations in data.items():
                                location_sub_group = location_group.create_group(ref_frame)
                                location_sub_group.attrs['description'] = 'the antenna locations in %s coordinates (units: meters and/or radians)' % (ref_frame,)

                                for antenna, location in antenna_locations.items():
                                    location_sub_group.create_dataset(antenna, data=location)
                        else:
                            ds_out = output_file.create_dataset(key, data=data)

                        #search first input_file containing this keys
                        #and copy all dataset attributes from the input_file to the output_file
                        try:
                            if ds_out:
                                input_file = next(f for f in input_files if key in f)
                                ds_in = input_file[key]

                                for attr_key, attr_value in ds_in.attrs.items():
                                    ds_out.attrs[attr_key] = attr_value
                        except StopIteration:
                            pass #no input file with key, so nothing to copy.

        fill_info_folder_from_parset(output_path)
    except Exception as e:
        logger.exception('combine_hypercubes: %s', e)
    finally:
        for h5file in input_files:
            h5file.close()

    logger.info('combine_hypercubes: finished combining %s h5 files into %s', len(input_paths), output_path)
    return output_path

DEFAULT_ALGO_NAME='scipy.cluster.hierarchical.single on visibility distance v1'

def _write_common_clustering_groups(h5_path, saps_dict, label=DEFAULT_ALGO_NAME):
    """
    helper method to write some common groups when writing clustering results into the h5_path

    :param str h5_path: path to the hdf5 file
    :param dict saps_dict: clustering results dict, see clusters parameter in write_clusters.
    :param str label: A name/label for this clustering result, for example 'my_clusterer_run_3'.
                      Multiple clustering results can be stored in the same h5 file, as long as the label is unique.
                      If the label was already present in the file, then it is overwritten.
                      The always present symlink 'latest' is updated to this clustering result.
    :return str: the name of the saps_group into which the non-common results can be written.
    """
    with SharedH5File(h5_path, "r+") as file:
        if 'clustering' in file:
            clustering_group = file['clustering']
        else:
            clustering_group = file.create_group('clustering')
            clustering_group.attrs['description'] = 'clustering results'

        if label == 'latest':
            raise ValueError('\'latest\' is a reserved label for a symlink to the actual latest clustering result.')

        if label in clustering_group:
            algo_group = clustering_group[label]
        else:
            algo_group = clustering_group.create_group(label)
            algo_group.attrs['description'] = 'clustering results for cluster method: %s' % label

        # always set/update the timestamp of this result
        algo_group.attrs['timestamp'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # update the 'latest' symlink to this label
        try:
            symlink = h5py.SoftLink('/clustering/' + label)
            if 'latest' in clustering_group:
                del clustering_group['latest']
            clustering_group['latest'] = symlink
        except RuntimeError:
            #softlink was already present, just continue.
            pass

        if 'saps' in algo_group:
            saps_group = algo_group['saps']
        else:
            saps_group = algo_group.create_group('saps')
            saps_group.attrs['description'] = 'clustering results are stored per sub array pointing'

        for sap_nr, sap_item in saps_dict.items():
            if str(sap_nr) not in saps_group:
                sap_group = saps_group.create_group(str(sap_nr))
                sap_group.attrs['description'] = 'clustering results for sub array pointing %d' % sap_nr

        return saps_group.name


def _delete_clustering_group_if_empty(h5_path, label):
    """
    helper method to delete an empty clustering group

    :param str h5_path: path to the hdf5 file
    :param str label: The name/label of the clustering group, for example 'my_clusterer_run_3'.
                      The always present symlink 'latest' is updated to the next latest clustering group result.
    """
    with SharedH5File(h5_path, "r+") as file:
        if 'clustering' in file:
            clustering_group = file['clustering']

            if label in clustering_group:
                algo_group = clustering_group[label]

                if not list(algo_group.keys()): #the algo groups is empty..., so delete it
                    del clustering_group[label]

                timestamped_algo_groups = [algo_group for algo_group in clustering_group.values() if 'timestamp' in algo_group.attrs]

                # update the 'latest' symlink to the latest result
                latest = datetime(0, 0, 0)
                for algo_group in timestamped_algo_groups:
                    if algo_group.attrs['timestamp'] >= latest:
                        clustering_group["latest"] = h5py.SoftLink('/clustering/' + algo_group.name)

def write_clusters(h5_path, clusters, label=DEFAULT_ALGO_NAME):
    """
    write the clusters into an h5 file.
    :param str h5_path: path to the h5 file
    :param dict clusters: the clusters results dict.
                                 { <sapnr>: { 'clusters': { <nr>: <list_of_baselines>, # for example: [('CS001', 'CS002), ('CS001', 'CS003')]
                                                          ... },
                                             ... },
                                 ... }
    :param str label: A name/label for this clustering result, for example 'my_clusterer_run_3'.
                      Multiple clustering results can be stored in the same h5 file, as long as the label is unique.
                      If the label was already present in the file, then it is overwritten.
                      The always present symlink 'latest' is updated to this clustering result.
    """
    logger.info('writing clusters to %s under label \'%s\'', h5_path, label)
    saps_group_name = _write_common_clustering_groups(h5_path, clusters, label=label)

    #add indirection level: cluster method (including run-timestamp)
    #include parameters and description
    with SharedH5File(h5_path, "r+") as file:
        saps_group = file[saps_group_name]
        for sap_nr, sap_clusters_dict in clusters.items():
            sap_group = saps_group[str(sap_nr)]

            clusters_group = sap_group.create_group('clusters')
            clusters_group.attrs['description'] = 'the clusters'

            sap_clusters = sap_clusters_dict['clusters']
            for cluster_nr in sorted(sap_clusters.keys()):
                cluster_baselines = sorted(sap_clusters[cluster_nr])
                logger.debug('writing %d baselines in cluster %s for sap %d to %s', len(cluster_baselines), cluster_nr, sap_nr, h5_path)

                ds = clusters_group.create_dataset(str(cluster_nr), data=cluster_baselines)
                ds.attrs['description'] = '%d baselines in cluster %d in sap %d' % (len(cluster_baselines), cluster_nr, sap_nr)
    logger.info('finished writing clusters to %s', h5_path)


def read_clusters(h5_path, label='latest'):
    """
    read the clusters from an h5 file.
    :param str h5_path: path to the h5 file
    :param str label: A name/label for this clustering result, for example 'my_clusterer_run_3', or the always present 'latest'.
    :return (dict, list): the clustering_results dict, and the clustering_results annotations list.

                  clustering_results = { <sapnr>: { 'clusters': { <nr>: <list_of_baselines>, # for example: [('CS001', 'CS002), ('CS001', 'CS003')]
                                                                ... },
                                                    'annotations': { <cluster_nr> : { 'annotation': <text>,
                                                                                      'user': <user>,
                                                                                      'timestamp: <datetime> },
                                                                     ... }
                                                   ... },
                                       ... }

                   annotations list = [ { 'annotation': <text>, 'user': <user>, 'timestamp: <datetime> },
                                        { 'annotation': <text>, 'user': <user>, 'timestamp: <datetime> },
                                        .... ]


    """
    result_clusters = {}
    result_annotations = []

    with SharedH5File(h5_path, "r") as file:
        if 'clustering' not in file:
            logger.debug('could not find any clustering results in %s', h5_path)
            return result_clusters, result_annotations

        clustering_group = file['clustering']

        if label not in clustering_group:
            logger.debug('could not find clusters for algorithm \'%s\' for in %s', label, h5_path)
            return result_clusters, result_annotations

        print(label)
        algo_group = clustering_group[label]

        logger.info('reading annotations for algorithm \'%s\', timestamp=\'%s\' from %s', label, algo_group.attrs.get('timestamp', '<unknown>'), h5_path)

        if 'annotations' in algo_group:
            for anno_nr, anno_ds in algo_group['annotations'].items():
                annotation = anno_ds[0]
                cluster_nr = anno_ds.attrs.get('cluster_nr')
                user = anno_ds.attrs.get('user')
                timestamp = anno_ds.attrs.get('timestamp')

                result_annotations.append({'annotation': annotation,
                                           'index': anno_nr,
                                           'user': user,
                                           'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')})

        saps_group = algo_group['saps']

        logger.info('reading clusters for algorithm \'%s\', timestamp=\'%s\' from %s', label, algo_group.attrs.get('timestamp', '<unknown>'), h5_path)

        for sap_nr, sap_dict in saps_group.items():
            sap_nr = int(sap_nr)
            sap_clusters_result = {}
            sap_clusters_annotations = {}
            sap_result = {'clusters': sap_clusters_result,
                          'annotations': sap_clusters_annotations }

            if 'clusters' in sap_dict:
                logger.debug('reading clusters for sap %d in %s', sap_nr, h5_path)

                result_clusters[sap_nr] = sap_result

                for cluster_nr in sorted(sap_dict['clusters'].keys()):
                    baselines = sap_dict['clusters'][cluster_nr][:]
                    cluster_nr = int(cluster_nr)
                    baselines = [(bl[0], bl[1]) for bl in baselines]
                    sap_clusters_result[cluster_nr] = baselines
                    logger.debug('read %d baselines in cluster %d in sap %d', len(baselines), cluster_nr, sap_nr)
            else:
                logger.debug('could not find clusters for sap %d in %s', sap_nr, h5_path)

            if 'annotations' in sap_dict:
                logger.debug('reading cluster annotations for sap %d in %s', sap_nr, h5_path)

                for anno_nr, anno_ds in sap_dict['annotations'].items():
                    try:
                        annotation = anno_ds[0]
                        cluster_nr = int(anno_ds.attrs.get('cluster_nr'))
                        logger.debug("%s %s", cluster_nr, type(cluster_nr))
                        user = anno_ds.attrs.get('user')
                        timestamp = anno_ds.attrs.get('timestamp')

                        if cluster_nr not in sap_clusters_annotations:
                            sap_clusters_annotations[cluster_nr] = []

                        sap_clusters_annotations[cluster_nr].append({'annotation': annotation,
                                                                     'index': anno_nr,
                                                                     'user': user,
                                                                     'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')})
                    except:
                        pass

                for cluster_nr, sap_clusters_annotation_list in sap_clusters_annotations.items():
                    logger.debug('read %d cluster annotations for cluster %d in sap %d', len(sap_clusters_annotation_list), cluster_nr, sap_nr)
            else:
                logger.debug('could not find cluster annotations for sap %d in %s', sap_nr, h5_path)

            logger.info('read %d clusters for sap %d from %s', len(sap_result), sap_nr, h5_path)
        logger.info('finised reading clusters from %s', h5_path)

    return result_clusters, result_annotations


def delete_clusters(h5_path, label=DEFAULT_ALGO_NAME):
    """
    delete the clustering results with the given label from the h5 file.
    :param str h5_path: h5_path to the h5 file
    :param str label: the name/label for of the clustering result, for example 'my_clusterer_run_3'.
                      The always present symlink 'latest' is updated to the next latest clustering result.
    """
    with SharedH5File(h5_path, "r+") as file:
        if 'clustering' in file:
            for name, group in file['clustering'].items():
                if label is None or name==label:
                    for sap_nr, sap_dict in group['saps'].items():
                        if 'clusters' in sap_dict:
                            logger.info('deleting clusters for sap %s in %s', sap_nr, h5_path)
                            del sap_dict['clusters']

    _delete_clustering_group_if_empty(h5_path, label)


def _add_annotation_to_group(annotations__parent_group, annotation, user=None, **kwargs):
    """
    add an annotation to the cluster in the file at h5_path, given by the clustering label, sap_nr, cluster_nr.
    :param str h5_path: h5_path to the h5 file
    :param str label: the label of the clustering results group
    :param int sap_nr: the sap number withing the clustering results group
    :param int cluster_nr: the cluster number withing the sap within the clustering results group
    :param str annotation: the annotation for this cluster (can be any free form text)
    :param str user: an optional user name
    """
    if 'annotations' in annotations__parent_group:
        annotations_group = annotations__parent_group['annotations']
    else:
        annotations_group = annotations__parent_group.create_group('annotations')
        annotations_group.attrs['description'] = 'annotations on this cluster'

    for seq_nr, ds in annotations_group.items():
        if ds[0] == annotation:
            if not 'cluster_nr' in kwargs or ('cluster_nr' in kwargs and ds.attrs['cluster_nr'] == kwargs['cluster_nr']):
                raise ValueError('annotation "%s" already exists' % (annotation,))

    seq_nr = max([int(x) for x in annotations_group.keys()])+1 if annotations_group.keys() else 0
    ds = annotations_group.create_dataset(str(seq_nr), (1,), h5py.special_dtype(vlen=str), annotation)
    ds.attrs['user'] = user if user else 'anonymous'
    ds.attrs['timestamp'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    for key, value in kwargs.items():
        ds.attrs[key] = value


def annotate_cluster(h5_path, label, sap_nr, cluster_nr, annotation, user=None):
    """
    add an annotation to the cluster in the file at h5_path, given by the clustering label, sap_nr, cluster_nr.
    :param str h5_path: h5_path to the h5 file
    :param str label: the label of the clustering results group
    :param int sap_nr: the sap number withing the clustering results group
    :param int cluster_nr: the cluster number withing the sap within the clustering results group
    :param str annotation: the annotation for this cluster (can be any free form text)
    :param str user: an optional user name
    """
    with SharedH5File(h5_path, "r+") as file:
        if 'clustering' in file:
            clustering_group = file['clustering']

            if label in clustering_group:
                algo_group = clustering_group[label]
                saps_group = algo_group['saps']

                if str(sap_nr) in saps_group:
                    sap_group = saps_group[str(sap_nr)]
                    _add_annotation_to_group(sap_group, annotation, user, cluster_nr=cluster_nr)

def delete_cluster_annotation(h5_path, sap_nr, cluster_nr, annotation_nr, label='latest'):
    """
    remove the annotation_nr'th annotation for the cluster in the file at h5_path, given by the clustering label, sap_nr, cluster_nr.
    :param str h5_path: h5_path to the h5 file
    :param str label: the label of the clustering results group
    :param int sap_nr: the sap number withing the clustering results group
    :param int cluster_nr: the cluster number withing the sap within the clustering results group
    :param str annotation_nr: the annotation number (index) to delete
    :param str label: the label of the clustering results group
    """
    with SharedH5File(h5_path, "r+") as file:
        if 'clustering' in file:
            clustering_group = file['clustering']

            if label in clustering_group:
                algo_group = clustering_group[label]
                saps_group = algo_group['saps']

                if str(sap_nr) in saps_group:
                    sap_group = saps_group[str(sap_nr)]
                    if 'annotations' in sap_group:
                        annotations_group = sap_group['annotations']
                        if 'annotations' in sap_group:
                            annotations_group = sap_group['annotations']
                            if str(annotation_nr) in annotations_group:
                                del annotations_group[str(annotation_nr)]

def annotate_clustering_results(h5_path, label, annotation, user=None):
    """
    add an annotation at top level for the entire file at h5_path.
    :param str h5_path: h5_path to the h5 file
    :param str label: the label of the clustering results group
    :param str annotation: the annotation for this cluster (can be any free form text)
    :param str user: an optional user name
    """
    with SharedH5File(h5_path, "r+") as file:
        if 'clustering' in file:
            clustering_group = file['clustering']

            if label in clustering_group:
                algo_group = clustering_group[label]
                _add_annotation_to_group(algo_group, annotation, user)


def annotate_file(h5_path, annotation, user=None):
    """
    add an annotation at top level for the entire file at h5_path.
    :param str h5_path: h5_path to the h5 file
    :param str annotation: the annotation for this cluster (can be any free form text)
    :param str user: an optional user name
    """
    with SharedH5File(h5_path, "r+") as file:
        _add_annotation_to_group(file, annotation, user)


def read_file_annotations(h5_path):
    """
    read the top level annotations on this file as a whole.
    :param str h5_path: path to the h5 file
    :return list: an annotations list with the top level annotations on this file as a whole.

                   annotations list = [ { 'annotation': <text>, 'user': <user>, 'timestamp: <datetime> },
                                        { 'annotation': <text>, 'user': <user>, 'timestamp: <datetime> },
                                        .... ]


    """
    result_annotations = []

    with SharedH5File(h5_path, "r") as file:
        if 'annotationss' in file:
            for anno_nr, anno_ds in file['annotations'].items():
                annotation = anno_ds[0]
                cluster_nr = anno_ds.attrs.get('cluster_nr')
                user = anno_ds.attrs.get('user')
                timestamp = anno_ds.attrs.get('timestamp')

                result_annotations.append({'annotation': annotation,
                                           'user': user,
                                           'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')})
    return result_annotations

def get_stations(h5_path):
    with SharedH5File(h5_path, "r+") as file:
        stations = set()
        for sap_dict in file['measurement/saps'].values():
            baselines = sap_dict['baselines'][:]
            for bl in baselines:
                stations.add(bl[0])
        return sorted(stations)

def read_info_from_hdf5(h5_path, read_data_info=True, read_parset_info=True):
    """
    Read basic info like Project, start/stoptime, stations, etc from h5 file.
    :param str h5_path: h5_path to the h5 file
    :param bool read_data_info: do/don't read data info (how many sap's, baselines, timestamps, subbands).
    :param bool read_parset_info: do/don't read info from the parset (Project, PI, name, start/stop time, etc).
    :return str: A human readable string with the requested info.
    """
    result = {}

    with SharedH5File(h5_path, "r") as file:
        need_to_fill_info_folder_from_parset = 'measurement/info' not in file

    if need_to_fill_info_folder_from_parset:
        # try to convert old style file with parsets only into new files with info.
        fill_info_folder_from_parset(h5_path)

    if read_data_info:
        result = read_hypercube(h5_path, read_visibilities=False, read_flagging=False)

    if read_parset_info:
        parset = read_hypercube_parset(h5_path)
        if parset:
            result['parset'] = parset

    file_annotations = read_file_annotations(h5_path)
    clusters, clustering_algorithm_annotations = read_clusters(h5_path)

    return create_info_string(result, h5_path, file_annotations, clusters, clustering_algorithm_annotations)


def create_info_string(data, h5_path=None, file_annotations=None, clusters=None, cluster_annotations=None):
    info = ''

    try:
        parset = data['parset']
        if h5_path:
            info += 'File                : ' + os.path.basename(h5_path) + '\n'
            try:
                with SharedH5File(h5_path, "r") as file:
                    info += 'File version        : ' + file['version'][0] + '\n'
            except IOError:
                pass

        info += 'Project             : ' + parset.getString('ObsSW.Observation.Campaign.name') + '\n'
        info += 'Project description : ' + parset.getString('ObsSW.Observation.Campaign.title') + '\n'
        info += 'Project PI          : ' + parset.getString('ObsSW.Observation.Campaign.PI') + '\n'
        info += 'Type                : ' + parset.getString('ObsSW.Observation.processSubtype') + '\n'
        info += 'SAS id              : ' + parset.getString('ObsSW.Observation.otdbID') + '\n'
        info += 'name                : ' + parset.getString('ObsSW.Observation.Scheduler.taskName') + '\n'
        info += 'start time (UTC)    : ' + parset.getString('ObsSW.Observation.startTime') + '\n'
        info += 'stop  time (UTC)    : ' + parset.getString('ObsSW.Observation.stopTime') + '\n'

        try:
            # try to import lofar.common.datetimeutils here and not at the top of the file
            # to make this hdf5_io module as loosly coupled to other lofar code as possible
            from lofar.common.datetimeutils import format_timedelta, parseDatetime
            info += 'duration            : ' + format_timedelta(parseDatetime(parset.getString('ObsSW.Observation.stopTime')) -
                                                                parseDatetime(parset.getString('ObsSW.Observation.startTime'))) + '\n'
        except ImportError:
            pass #just continue

        if 'observation' in parset.getString('ObsSW.Observation.processSubtype','').lower():
            info += '#Stations           : ' + str(len(parset.getStringVector('ObsSW.Observation.VirtualInstrument.stationList'))) + '\n'
            info += 'Stations            : ' + ','.join(sorted(parset.getStringVector('ObsSW.Observation.VirtualInstrument.stationList'))) + '\n'
            info += 'antenna array       : ' + parset.getString('ObsSW.Observation.antennaArray') + '\n'
    except:
        #parset info not available
        pass

    if file_annotations:
        for i, anno in enumerate(file_annotations):
            info += 'annotation[%02d]      : \'%s\', by \'%s\' at \'%s\'\n' % (i, anno['annotation'], anno['user'], anno['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))

    if 'saps' in data:
        for sap_nr, sap_dict in data['saps'].items():
            info += 'data                : sap: %s, #baselines: %s, #timestamps: %s, #subbands: %s, #polarizations: %s' % (
                sap_nr, len(sap_dict['baselines']), len(sap_dict['timestamps']), len(sap_dict['subbands']), len(sap_dict['polarizations'])) + '\n'

    if clusters:
        for sap_nr in sorted(clusters.keys()):
            sap_dict = clusters[sap_nr]
            sap_cluster_dict = sap_dict['clusters']
            info += 'clusters            : sap: %s, #clusters: %s, cluster sizes: %s' % (
                sap_nr, len(sap_cluster_dict), ', '.join([str(len(sap_cluster_dict[c_nr])) for c_nr in sorted(sap_cluster_dict.keys())])) + '\n'

            sap_cluster_annotation_dict = sap_dict.get('annotations', {})
            for sap_cluster_nr in sorted(sap_cluster_annotation_dict.keys()):
                sap_cluster_annotations = sap_cluster_annotation_dict[sap_cluster_nr]
                for sap_cluster_annotation in sap_cluster_annotations:
                    info += 'annotations         : sap: %d cluster: %d : %s %s "%s"\n' % (sap_nr, sap_cluster_nr,
                                                                      sap_cluster_annotation.get('user', '<unknown>'),
                                                                      sap_cluster_annotation.get('timestamp', '<unknown>'),
                                                                      sap_cluster_annotation.get('annotation', '<unknown>'))

    return info


def fill_info_folder_from_parset(h5_path):
    try:
        logger.info('fill_info_folder_from_parset for %s', h5_path)
        parset = read_hypercube_parset(h5_path)

        if parset is not None:
            with SharedH5File(h5_path, "r+") as file:
                # remove previous info if present
                if 'measurement/info' in file:
                    del file['measurement/info']

                info_group = file.create_group('measurement/info')
                info_group.attrs['description'] = 'Meta information about the measurement'

                for name, key in [('project', 'Campaign.name'),
                                  ('project_description', 'Campaign.title'),
                                  ('PI', 'Campaign.PI'),
                                  ('type', 'processType'),
                                  ('subtype', 'processSubtype'),
                                  ('SAS_id', 'Campaign.otdbID'),
                                  ('antenna_array', 'antennaArray'),
                                  ('name', 'Scheduler.taskName')]:
                    ps_key = 'ObsSW.Observation.' + key
                    ps_value = parset.getString(ps_key, '<unknown>')
                    info_group.create_dataset(name, (1,), h5py.special_dtype(vlen=str), [ps_value.encode('utf-8')])

                try:
                    # try to import lofar.common.datetimeutils here and not at the top of the file
                    # to make this hdf5_io module as loosly coupled to other lofar code as possible
                    from lofar.common.datetimeutils import format_timedelta, parseDatetime, totalSeconds
                    start_time = parset.getString('ObsSW.Observation.startTime')
                    stop_time = parset.getString('ObsSW.Observation.stopTime')
                    duration = parseDatetime(stop_time) - parseDatetime(start_time)
                    info_group.create_dataset('start_time', (1,), h5py.special_dtype(vlen=str), [start_time.encode('utf-8')])
                    info_group.create_dataset('stop_time', (1,), h5py.special_dtype(vlen=str), [stop_time.encode('utf-8')])
                    ds = info_group.create_dataset('duration', data=[totalSeconds(duration)])
                    ds.attrs['description'] = 'duration in seconds'
                except (ImportError, RuntimeError, ValueError) as e:
                    logger.warning('Could not convert start/end time and/or duration in fill_info_folder_from_parset for %s. error: %s', h5_path, e)
    except Exception as e:
        logger.error('Error while running fill_info_folder_from_parset: %s', e)

def read_info_dict(h5_path):
    ''' read the info about the observation/pipeline from the h5 file given by h5_path.
    :param str h5_path: h5_path to the h5 file
    :return: a dict with the info about the observation/pipeline in native python types, like:
            {'PI': 'my_PI',
             'SAS_id': 'my_id',
             'duration': datetime.timedelta(0, 3600),
             'name': 'my_observation_name',
             'project': 'my_project_name',
             'project_description': 'my_project_description',
             'antenna_array': 'LBA',
             'start_time': datetime.datetime(2018, 6, 11, 11, 0),
             'stop_time': datetime.datetime(2018, 6, 11, 12, 0),
             'type': 'my_process_subtype'} '''

    with SharedH5File(h5_path, "r", timeout=10) as file:
        need_to_fill_info_folder_from_parset = 'measurement/info' not in file

    if need_to_fill_info_folder_from_parset:
        fill_info_folder_from_parset(h5_path)

    with SharedH5File(h5_path, "r", timeout=10) as file:
        info_dict = {}
        if 'measurement/info' in file:
            for k, v in file['measurement/info'].items():
                k = str(k)
                v = v[0]
                info_dict[k] = v

                if k == 'start_time' or k == 'stop_time':
                    # try to import lofar.common.datetimeutils here and not at the top of the file
                    # to make this hdf5_io module as loosly coupled to other lofar code as possible
                    try:
                        from lofar.common.datetimeutils import parseDatetime
                        info_dict[k] = parseDatetime(v)
                    except ImportError:
                        pass
                elif k == 'duration':
                    info_dict[k] = timedelta(seconds=v)

        return info_dict

def read_SAP_targets(h5_path):
    """reads the SAP targets from the parset
    :param str h5_path: h5_path to the h5 file
    :return: dict of SAP_nr to target name.
    """

    beam_dict = {}

    #TODO: use normal parset lib instead of error prone string parsing
    try:
        parset_str = read_hypercube_parset(h5_path, as_string=True)
        if parset_str:
            lines = parset_str.splitlines()
            beam_lines = [l for l in lines if 'Observation.Beam[' in l and '.target' in l]
            for line in beam_lines:
                parts = line.partition('=')
                beam_nr = int(parts[0][parts[0].index('[')+1: parts[0].index(']')])
                beam_dict[beam_nr] = parts[2]

    except Exception as e:
        logger.error(e)

    return beam_dict


