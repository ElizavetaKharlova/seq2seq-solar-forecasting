from operator import itemgetter
import pickle
import random

secure_random = random.SystemRandom()

def create_offline_profile(interval_s, units, time_start, load, gen):
    offline_profile = {
        'load':{
            'interval_s': interval_s,
            'units': units,
            'time_start': time_start,
            'profile': load
        },
        'gen_solar': {
            'interval_s': interval_s,
            'units': units,
            'time_start': time_start,
            'profile': gen
        }
    }
    return offline_profile

def dump_pickle(filename, object_to_pickle):
    import gzip
    with open(filename, 'wb') as handle:
        pickle.dump(object_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # pass

def import_pickle(filename):
    import gzip
    import pickle
    with open(filename, 'rb') as handle:
        p_obj = pickle.load(handle)
    return p_obj

def dump_zp(filename, object):
    import gzip, pickle
    try:
        f = gzip.open(filename + '.zp', 'wb')
        pickle.dump(object, f, 1)
        f.close()
        return True
    except:
        return False

def import_zp(filename):
    import gzip, pickle
    f = gzip.open(filename + '.zp', 'rb')
    p_obj = pickle.load(f)
    f.close()
    return p_obj

def import_profile(pickle_path_name, time_start=None, time_end=None):
    from datetime import datetime
    offline_profile = import_zp(pickle_path_name)

    if 'load' not in offline_profile:
        return None

    time_start_dt = None
    time_start_index = 0
    time_end_index = len(offline_profile['load']['profile'])-1

    if time_start is not None:
        time_start_dt = datetime.strptime(time_start, '%Y-%m-%d%H:%M:%S')
        time_start_index = int((time_start_dt - offline_profile['load']['time_start']).total_seconds() / 60)

    if time_end is not None:
        time_end_dt = datetime.strptime(time_end, '%Y-%m-%d%H:%M:%S')
        time_end_index = int((time_end_dt - offline_profile['load']['time_start']).total_seconds() / 60)

    offline_profile['load']['time_start'] = time_start_dt
    offline_profile['load']['profile'] = offline_profile['load']['profile'][time_start_index:time_end_index]

    if 'gen' in offline_profile:
        offline_profile['gen']['time_start'] = time_start_dt
        offline_profile['gen']['profile'] = offline_profile['gen']['profile'][time_start_index:time_end_index]

    return offline_profile

# for _ in range(1, 11):
#     profile = import_profile('../../Data/raw/apartment/2016/Apt' + str(_) + '_2016.csv', 0, 1)
#     dump_pickle('../../Data/pkl/Apt' + str(_) + '_2016.pickle', profile)