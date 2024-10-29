import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from t3d.Profiles import FluxProfile, GridProfile
from collections import defaultdict
from t3d.Logbook import warn, errr


class Import():

    def __init__(self, inputs, grid):
        self.grid = grid

        # read import parameters from input file
        import_parameters = inputs.get('import', {})
        self.type = import_parameters.get('type', None)
        self.file = import_parameters.get('file', '')
        if self.type is None:
            self.imported = None
        elif self.type == "transp":
            transp_time = import_parameters.get('transp_time', 0.0)
            self.imported = TranspReader(self.file, time=transp_time, flux_label=grid.flux_label)
        elif self.type == "trinity":
            trinity_frame = import_parameters.get('frame', -1)
            self.imported = TrinityReader(grid, self.file, trinity_frame)
        elif self.type == "ascii" or self.type == "text" or self.type == "txt" or self.type == "columns":
            columns = import_parameters.get('columns', [])  # list of keys corresponding to columns in file
            divide_by = import_parameters.get('divide_by', [])  # list of factors to divide columns by to get expected units
            if len(divide_by) == 0:
                divide_by = [1 for c in columns]
            assert len(divide_by) == len(columns), "import.columns and import.divide_by lists must be same length"
            self.imported = AsciiColumnReader(grid, self.file, columns=columns, divide_by=divide_by)
        else:
            raise ValueError('Invalid import type')

    def get(self, key):
        return self.imported.data[key]

    def keys(self):
        return [k for k in self.imported.data.keys()]

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, interpolate=True, n_edge=None, norms=None):
        rho, n = self.imported.get_density(key=key, species_mass=species_mass, species_charge=species_charge, tag=tag, norms=norms)

        # interpolate onto Trinity grid
        if interpolate:
            f = interp1d(rho, n, bounds_error=False, fill_value='extrapolate')
            n = f(self.grid.rho)

        if n_edge:
            n[-1] = n_edge

        return n

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, interpolate=True, T_edge=None, norms=None):
        rho, T = self.imported.get_temperature(key=key, species_mass=species_mass, species_charge=species_charge, tag=tag, norms=norms)

        # interpolate onto Trinity grid
        if interpolate:
            f = interp1d(rho, T, bounds_error=False, fill_value='extrapolate')
            T = f(self.grid.rho)

        if T_edge:
            T[-1] = T_edge

        return T

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, interpolate=True, norms=None):
        rho, Sn = self.imported.get_density_source(keys=keys, species_mass=species_mass, species_charge=species_charge, tag=tag, norms=norms)

        Sn_trin = np.zeros(len(self.grid.rho))
        # interpolate onto Trinity grid
        if interpolate and self.type != 'trinity':
            f = interp1d(rho, Sn, bounds_error=False, fill_value='extrapolate')
            # cell average
            for i in np.arange(len(self.grid.rho)):
                Sn_trin[i] = integrate.quad(f, self.grid.rho[i]-self.grid.drho/2, self.grid.rho[i]+self.grid.drho/2)[0]/self.grid.drho
        else:
            Sn_trin = Sn

        return Sn_trin

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, interpolate=True, norms=None):
        rho, Sp = self.imported.get_pressure_source(keys=keys, species_mass=species_mass, species_charge=species_charge, tag=tag, norms=norms)

        Sp_trin = np.zeros(len(self.grid.rho))
        # interpolate onto Trinity grid
        if interpolate and self.type != 'trinity':
            f = interp1d(rho, Sp, bounds_error=False, fill_value='extrapolate')
            # cell average
            for i in np.arange(len(self.grid.rho)):
                Sp_trin[i] = integrate.quad(f, self.grid.rho[i]-self.grid.drho/2, self.grid.rho[i]+self.grid.drho/2)[0]/self.grid.drho
        else:
            Sp_trin = Sp

        return Sp_trin

    def get_geometry(self, geo, interpolate=True):

        self.imported.get_geometry(geo, self.grid, interpolate=interpolate)

    def get_Zeff(self, key=None, interpolate=True):
        rho, Z = self.imported.get_Zeff(key=key)
        if not hasattr(Z, '__len__'):
            Z = np.ones(len(rho))*Z

        # interpolate onto Trinity grid
        if interpolate:
            f = interp1d(rho, Z, bounds_error=False, fill_value='extrapolate')
            Z = f(self.grid.rho)

        return Z


class TrinityReader():
    '''
    Read data from a Trinity or Trinity-like .npy file containing a dictionary
    '''

    def __init__(self, grid, fin, frame=None):

        self.grid = grid
        self.frame = frame
        self.data = np.load(fin, allow_pickle=True).tolist()

        # backward compatibility (5/2024): older t3d files use deprecated rho_axis
        if self.data['grid']['rho'] is None:
            self.data['grid']['rho'] = self.data['grid']['rho_axis']

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key:
            n = np.array(self.data[key])[self.frame]
            return self.data['grid']['rho'], n
        elif tag:
            n = np.array(self.data[f'n_{tag}'])[self.frame]
            return self.data['grid']['rho'], n
        else:
            return None, None

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key:
            T = np.array(self.data[key])[self.frame]
            return self.data['grid']['rho'], T
        elif tag:
            T = np.array(self.data[f'T_{tag}'])[self.frame]
            return self.data['grid']['rho'], T
        else:
            return None, None

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if keys:
            for key in keys:
                Sn = np.array(self.data[key])[self.frame]
                return self.data['grid']['rho'], Sn
        elif tag:
            Sn = np.array(self.data[f'Sn_aux_{tag}'])[self.frame]
            return self.data['grid']['rho'], Sn
        else:
            return None, None

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if keys:
            for key in keys:
                Sp = np.array(self.data[key])[self.frame]
                return self.data['grid']['rho'], Sp
        elif tag:
            Sp = np.array(self.data[f'Sp_aux_{tag}'])[self.frame]
            return self.data['grid']['rho'], Sp
        else:
            return None, None


class AsciiColumnReader():
    '''
    Read generic columned data from an ascii text file
    '''

    def __init__(self, grid, fin, columns=[], divide_by=[]):

        self.grid = grid
        self.columns = columns
        self.divide_by = divide_by
        self.data = defaultdict(list)
        numlines = 0
        with open(fin) as f:
            for line in f:
                if line[0] == '#':
                    continue
                data = line.split()
                for i, k in enumerate(columns):
                    self.data[k].append(float(data[i])/divide_by[i])
                numlines += 1

        if 'rho' not in self.data.keys():
            warn("A radial coordinate 'rho' was not specified in the import.columns list. Assuming rho = [0, ..., 1]")
            self.data['rho'] = np.linspace(0, 1, num=numlines)

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key:
            n = np.array(self.data[key])
            return self.data['rho'], n
        elif tag:
            n = np.array(self.data[f'n_{tag}'])
            return self.data['rho'], n
        else:
            return None, None

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key:
            T = np.array(self.data[key])
            return self.data['rho'], T
        elif tag:
            T = np.array(self.data[f'T_{tag}'])
            return self.data['rho'], T
        else:
            return None, None

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if keys:
            keys = [keys] if isinstance(keys, str) else keys
            for key in keys:
                Sn = np.array(self.data[key])
                return self.data['rho'], Sn/norms.Sn_ref_SI20
        elif tag:
            Sn = np.array(self.data[f'Sn_aux_{tag}'])
            return self.data['rho'], Sn/norms.Sn_ref_SI20
        else:
            return None, None

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if keys:
            keys = [keys] if isinstance(keys, str) else keys
            for key in keys:
                Sp = np.array(self.data[key])
                return self.data['rho'], Sp/norms.P_ref_MWm3
        elif tag:
            Sp = np.array(self.data[f'Sp_aux_{tag}'])
            return self.data['rho'], Sp/norms.P_ref_MWm3
        else:
            return None, None


class TranspReader():
    '''
    Read data from a TRANSP u-file, which can be used for initial profiles, sources, geometry etc
    '''

    def __init__(self, fin, time=None, flux_label='torflux'):

        self.time = time
        self.flux_label = flux_label

        if fin[-3:] == '.nc' or fin[-4:] == '.cdf':
            self.read_nc(fin, time=self.time)
        else:
            self.read_ascii(fin, time=self.time)

    def read_nc(self, fin2d, time=None):
        assert False, 'TRANSP netcdf reader not yet implemented.'

    def read_ascii(self, fin2d, time=None):
        fin0d = fin2d.replace('2d', '0d')
        fin1d = fin2d.replace('2d', '1d')

        self.meta = {}
        self.data = {}

        # read 0d data
        with open(fin0d) as f:
            datain = f.readlines()

        label = datain[0].strip().split(',')
        value = datain[1].strip().split(',')

        for j in np.arange(len(label)):

            self.data[label[j]] = value[j]

        # read 1d (time) data
        with open(fin1d) as f:
            datain = f.readlines()

        block_length1d = 0
        for d in datain:
            block_length1d += 1
            if d[:10] == '**********':
                block_length1d += 1
                break

        N_blocks = len(datain) // block_length1d
        for j in np.arange(N_blocks):

            start = j*block_length1d
            end = start + block_length1d

            block = datain[start:end]
            obj = TData1D(block, block_length1d, time=time)

            self.meta[obj.label] = obj
            self.data[obj.label] = obj.data

        # read 2d (time, rho) data
        with open(fin2d) as f:
            datain = f.readlines()

        block_length2d = 0
        for d in datain:
            block_length2d += 1
            if d[:10] == '**********':
                block_length2d += 1
                break

        N_blocks = len(datain) // block_length2d
        for j in np.arange(N_blocks):

            start = j*block_length2d
            end = start + block_length2d

            block = datain[start:end]
            obj = TData2D(block, block_length2d, time=time)

            self.meta[obj.label] = obj
            self.data[obj.label] = obj.data

    def find_ion_species_index(self, species_mass, species_charge=None):
        masses = [float(self.data[f'NM{i}A']) for i in np.arange(1, 10)]
        charges = [float(self.data[f'NM{i}Z']) for i in np.arange(1, 10)]

        if all(m < 0 for m in masses):
            masses = [2.0]
            charges = [1.0]

        try:
            index = masses.index(species_mass)
        except RuntimeError:
            errr(f'Error: species with mass {species_mass} not found in TRANSP file.')

        if species_charge:
            assert species_charge == charges[index]

        return index + 1  # 1-based indexing

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key:
            rhotor, n = self.meta[key].rho, self.data[key]
        elif species_charge == -1:
            rhotor, n = self.meta['NE'].rho, self.data['NE']
        else:
            try:
                index = self.find_ion_species_index(species_mass, species_charge)
                rhotor, n = self.meta[f'NM{index}'].rho, self.data[f'NM{index}']
            except:
                rhotor, n = self.meta['NE'].rho, self.data['NE']

        if self.flux_label == "rminor":
            try:
                rmin = self.data['RMINOR']/self.data['AMIN']
            except:
                rmin = self.data['RMINOR']/np.expand_dims(self.data['AMIN'], axis=1)
            rho = rmin
        else:
            rho = rhotor

        # convert to Trinity units, 10^20 m^-3
        n = n*1e-20

        return rho, n

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key:
            rhotor, T = self.meta[key].rho, self.data[key]
        elif species_charge == -1:
            rhotor, T = self.meta['TE'].rho, self.data['TE']
        else:
            rhotor, T = self.meta['TI'].rho, self.data['TI']

        if self.flux_label == "rminor":
            try:
                rmin = self.data['RMINOR']/self.data['AMIN']
            except:
                rmin = self.data['RMINOR']/np.expand_dims(self.data['AMIN'], axis=1)
            rho = rmin
        else:
            rho = rhotor

        # convert to Trinity units, keV
        T = T*1e-3

        return rho, T

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        _, Sn_tot = self.get_density(key='NE')
        Sn_tot = 0*Sn_tot

        # smart defaults
        if keys is None:
            if species_charge == -1:
                keys = ['SBE', 'SNBIE', 'DNER']
            elif self.find_ion_species_index(species_mass, species_charge) == 1:
                keys = ['SNBII', 'SWALL', 'DNER']

        for key in keys:
            if key in self.data.keys():
                rhotor, Sn = self.meta[key].rho, self.data[key]
                if key == 'DNER':
                    Sn = -1*Sn
                Sn_tot += Sn

        if self.flux_label == "rminor":
            try:
                rmin = self.data['RMINOR']/self.data['AMIN']
            except:
                rmin = self.data['RMINOR']/np.expand_dims(self.data['AMIN'], axis=1)
            rho = rmin
        else:
            rho = rhotor

        # convert to Trinity units
        # TRANSP sources are in 1/(m^3 s)
        if norms:
            Sn_tot = Sn_tot*1e-20/norms.Sn_ref_SI20

        return rho, Sn_tot

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        _, Sp_tot = self.get_density(key='NE')
        Sp_tot = 0*Sp_tot

        # smart defaults
        if keys is None:
            if species_charge == -1:
                keys = ['QNBIE', 'QICRHE', 'QECHE', 'QLHE', 'QIBWE', 'QWALLE', 'QOHM', 'DWER']
            elif self.find_ion_species_index(species_mass, species_charge) == 1:
                keys = ['QNBII', 'QICRHI', 'QECHI', 'QLHI', 'QIBWI', 'QWALLI', 'DWIR']

        for key in keys:
            if key in self.data.keys():
                rhotor, Sp = self.meta[key].rho, self.data[key]
                if key == 'DWER' or key == 'DWIR':
                    Sp = -1*Sp
                Sp_tot += Sp

        if self.flux_label == "rminor":
            try:
                rmin = self.data['RMINOR']/self.data['AMIN']
            except:
                rmin = self.data['RMINOR']/np.expand_dims(self.data['AMIN'], axis=1)
            rho = rmin
        else:
            rho = rhotor

        # convert to Trinity units
        # TRANSP sources are in W/m^3
        if norms:
            Sp_tot = Sp_tot*1e-6/norms.P_ref_MWm3

        return rho, Sp_tot

    def get_geometry(self, geo, grid, interpolate=True):

        # read in geometry quantities
        # 0d quantities
        aminor = self.data['AMIN']
        B_T = self.data['BT']
        rgeo = self.data['RGEO']/aminor

        # 1d quantities, on rhotor grid
        rhotor = self.meta['RMINOR'].rho
        rminor = self.data['RMINOR']/aminor
        if self.flux_label == 'rminor':
            rho = rminor
            drho_drhotor = np.gradient(rminor, rhotor)
        elif self.flux_label == 'torflux':
            rho = rhotor
            drho_drhotor = np.ones(rhotor.shape)

        rmajor = self.data['RMAJOR']/aminor
        qsf = self.data['Q']
        kappa = self.data['KAPPAR']
        delta = np.arcsin(self.data['DELTAR'])
        area = self.data['SURF']/aminor**2

        # dF/drho = dF/drhotor * drhotor/drho = dF/drhotor / (drho/drhotor)
        rmajor_prime = np.gradient(rmajor, rhotor)/drho_drhotor
        kappa_prime = np.gradient(kappa, rhotor)/drho_drhotor
        delta_prime = np.gradient(delta, rhotor)/drho_drhotor
        shat = np.gradient(qsf, rhotor)/drho_drhotor*rho/qsf
        grho = self.data['GRHO1']*aminor*drho_drhotor

        geo.Btor = B_T
        geo.a_minor = aminor
        geo.AspectRatio = rgeo

        if interpolate:
            f = interp1d(rho, area, bounds_error=False, fill_value='extrapolate')
            geo.area = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, grho, bounds_error=False, fill_value='extrapolate')
            geo.grho = FluxProfile(f(grid.midpoints), grid)
            geo.grho_grid = GridProfile(f(grid.rho), grid)

            f = interp1d(rho, rhotor, bounds_error=False, fill_value='extrapolate')
            geo.rhotor = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, rminor, bounds_error=False, fill_value='extrapolate')
            geo.rminor = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, rmajor, bounds_error=False, fill_value='extrapolate')
            geo.R_geo = FluxProfile(f(grid.midpoints), grid)  # This could be used in a normalized Miller model for R(rho)

            f = interp1d(rho, rmajor_prime, bounds_error=False, fill_value='extrapolate')
            geo.rmajor_prime = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, qsf, bounds_error=False, fill_value='extrapolate')
            geo.qsf = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, shat, bounds_error=False, fill_value='extrapolate')
            geo.shat = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, kappa, bounds_error=False, fill_value='extrapolate')
            geo.kappa = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, delta, bounds_error=False, fill_value='extrapolate')
            geo.delta = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, kappa_prime, bounds_error=False, fill_value='extrapolate')
            geo.kappa_prime = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, delta_prime, bounds_error=False, fill_value='extrapolate')
            geo.delta_prime = FluxProfile(f(grid.midpoints), grid)

    def get_Zeff(self, key=None):
        if key:
            rhotor, Z = self.meta[key].rho, self.data[key]
        else:
            try:
                rhotor, Z = self.meta['ZEFFR'].rho, self.data['ZEFFR']
            except:
                rhotor, Z = self.meta['NE'].rho, self.data['ZEFF']

        if self.flux_label == "rminor":
            try:
                rmin = self.data['RMINOR']/self.data['AMIN']
            except:
                rmin = self.data['RMINOR']/np.expand_dims(self.data['AMIN'], axis=1)
            rho = rmin
        else:
            rho = rhotor

        return rho, Z


class TData():
    @staticmethod
    def read_header(block):

        header_dict = {}
        for line in block:
            val, key = line.strip().split(';')
            tag = key.split('-')[1]  # get piece in between "-tag-"
            header_dict[tag] = val

        return header_dict

    @staticmethod
    def parse_line(line, j=13):

        N = len(line)
        return [line[idx:idx + j] for idx in range(0, N, j)]

    @staticmethod
    def parse(line, j=13):

        i = 1
        num = line[i:i+j]
        arr = []
        while num.strip():
            arr.append(num)

            # set next round
            i += j
            num = line[i:i+j]

        return arr

    @staticmethod
    def read_block(block):

        data = []
        for line in block:
            try:
                # reads most lines
                row = np.array(line.strip().split(' '), float)
            except:
                # reads lines with consectutive negs
                row = np.array(TData.parse(line), float)
                # np.array(parse_line(line[1:-1]), float)

            for n in row:
                data.append(n)

        return np.array(data)


class TData1D(TData):
    def __init__(self, datain, block_length, time=None, header_length=7, footer_length=3, label_length=15):

        start = header_length
        header = datain[:start]
        head = TData.read_header(header)
        len_data = int(head['# OF PTS'])

        end = block_length - footer_length
        block = datain[start:end]

        data = TData.read_block(block)

        var_label = head['DEPENDENT VARIABLE LABEL']

        label = var_label[:label_length].strip()
        units = var_label[label_length:].strip()

        xaxis = head['INDEPENDENT VARIABLE LABEL'].strip()

        x = data[:len_data]
        y = data[len_data:]

        self.len_data = len_data
        self.header = head
        self.label = label
        self.units = units
        self.xaxis = xaxis
        self.times = x
        self.data = y

        if time:
            # find time index with time closest to target time
            tid = (np.abs(x - time)).argmin()
            self.data = self.data[tid]
            self.time = self.times[tid]


class TData2D(TData):
    def __init__(self, datain, block_length, time=None, header_length=9, footer_length=3, label_length=15):

        start = header_length
        end = block_length - footer_length

        header = datain[:start]
        block = datain[start:end]

        head = TData.read_header(header)
        data = TData.read_block(block)

        xlen_data = int(head['# OF X PTS'])
        ylen_data = int(head['# OF Y PTS'])
        var_label = head['DEPENDENT VARIABLE LABEL']

        label = var_label[:label_length].strip()
        units = var_label[label_length:].strip()

        xaxis = head['INDEPENDENT VARIABLE (X) LABEL'].strip()
        yaxis = head['INDEPENDENT VARIABLE (Y) LABEL'].strip()

        x = data[:xlen_data]
        y = data[xlen_data:xlen_data + ylen_data]
        f = data[xlen_data+ylen_data:]

        self.xlen_data = xlen_data
        self.ylen_data = ylen_data
        self.header = head
        self.label = label
        self.units = units
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.rho = x
        self.times = y
        self.data = np.reshape(f, (ylen_data, xlen_data))

        if time:
            # find time index with time closest to target time
            tid = (np.abs(y - time)).argmin()
            self.data = self.data[tid, :]
            self.time = self.times[tid]
