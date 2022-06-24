# coding=utf-8
from pi3diamond import pi3d
import datetime
import numpy as np
import os
import UserScripts.helpers.sequence_creation_helpers as sch; reload(sch)
import UserScripts.helpers.shared as shared; reload(shared)
import multi_channel_awg_seq as MCAS; reload(MCAS)
import UserScripts.helpers.snippets_awg as sna; reload(sna)
import UserScripts.helpers.shared as ush;reload(ush)
from qutip_enhanced import *
import AWG_M8190A_Elements as E
import pym8190a.elements as e
from collections import OrderedDict
import AWG_M8190A_Elements as E

seq_name = os.path.basename(__file__).split('.')[0]
nuclear = sch.create_nuclear(__file__)
with open(os.path.abspath(__file__).split('.')[0] + ".py", 'r') as f:
    meas_code = f.read()

__TAU_HALF__ = 2*192/12e3
__SAMPLE_FREQUENCY__ = e.__SAMPLE_FREQUENCY__

ael = 1.0

def ret_ret_mcas(pdc):
    def ret_mcas(current_iterator_df):
        sequence_name = 'Electron_t2_red'

        mcas = MCAS.MultiChSeq(name=sequence_name, ch_dict={'2g': [1, 2], 'ps': [1]})

        def pi(ms=-1):

            if ms == 1:
                pi_ = pi3d.tt.rp('e_rabi_ou350deg-90-R', amp=1.0).pi
                freq_pi = pi3d.tt.mfl({'14N': [0]}, ms_trans='+1')

            elif ms == -1:
                freq_pi = pi3d.tt.mfl({'14N': [0]}, ms_trans='-1')
                pi_ = pi3d.tt.rp('e_rabi_ou350deg-90-L', amp=1.0).pi

            else:
                freq_pi = pi3d.tt.mfl({'14N': [0]}, ms_trans='-1')
                pi_ = pi3d.tt.rp('e_rabi_ou350deg-90-L', amp=1.0).pi

            erabi(freq=freq_pi, length=pi_ , amp = 1.0, phase=0)
            


        def erabi(freq, length, amp, phase=0.0):
            sna.electron_rabi(mcas,
                              name='electron rabi',
                              length_mus=length,
                              amplitudes=[amp],
                              frequencies=freq,
                              phases=np.rad2deg(phase),
                              new_segment=False,
                              mixer_deg=-90,
                              )

        def waveform(seq, ms = '+1', new_wave = True):
            if new_wave:
                mcas.start_new_segment('waveform')
            #sna.polarize(mcas, new_segment=False)
            mw = seq.times_fields_aphi('mw')
            wait = seq.times_fields_aphi('wait')
            freq_dd = pi3d.tt.mfl({'14N': [0]}, ms_trans= ms )
            ending = {'-1':'L', '+1':'R'}[ms]
            
            for step in seq.sequence_steps:
                idx = int(step[1]) - 1
                if step[0] == 'mw':

                    amp = pi3d.tt.rp('e_rabi_ou350deg-90-{}'.format(ending), period=1 / mw[idx, 1]).amp
                    print(mw[idx], ms, ending, amp, freq_dd)
                    erabi(freq=freq_dd, length=mw[idx, 0],
                          amp=amp, phase=mw[idx, 2])
                elif step[0] == 'wait':
                    mcas.asc(length_mus=wait[idx, 0])

        def dd(ms = '-1', phase_last = 0, nw = True):
            pi2x = sc.Rabi(t_rabi=0.25 * rabi_period, omega=1.0 / rabi_period, phase=0.0, control_field='mw')
            pi2_2 = sc.Rabi(t_rabi=0.25 * rabi_period, omega=1.0 / rabi_period, phase=phase_last,
                            control_field='mw')
            dd = sc.DD(dd_type='{}_{}'.format(_I_['n_rep_dd'], _I_['ddt']), rabi_period=_I_['rabi_period'],
                       tau = _I_['tau'])#total_tau=_I_['total_tau'])
            seq = sc.Concatenated([pi2x, dd, pi2_2], controls=['mw', 'wait'])
            waveform(ms= ms, seq = seq, new_wave=nw)



        for idx, _I_ in current_iterator_df.iterrows():

            rabi_period = _I_['rabi_period']

            mcas.start_new_segment('start_sequence')
            mcas.asc(length_mus=0.1)  # Starting... histogram 0

            mcas.asc(length_mus=1.0, name = 'initial_delay')
            mcas.asc(length_mus=3.0, green = True, name='Green')
            mcas.asc(length_mus=1.0)
            #mcas.asc(aom_A1=True, length_mus=100, name = 'A1_init')  # Init NV with A1 laser (about 1-3 µs). This step can be skipped for the very first tests #as the green laser will also intialise somehow.
            mcas.asc(length_mus=2.0)

            dd(ms='-1',phase_last=np.pi*0.5, nw = False)
            mcas.asc(length_mus=_I_['T_C1'])


            ## Flip the DQ populations
            pi(ms=-1)
            mcas.asc(length_mus = 0.1)
            pi(ms=1)
            mcas.asc(length_mus=0.1)
            pi(ms=-1)
            # Second sweep..
            mcas.asc(length_mus=_I_['T_C2'])
            dd(ms='+1',phase_last=np.pi*0.5+_I_['phase_pi2_2'], nw = False)
            #mcas.asc(length_mus=100.0-_I_['T_C'])

            freq = pi3d.tt.mfl({'14N': [0]}, ms_trans=_I_['ms'])
            sna.ssr(mcas, frequencies=freq, wait_dur=0.0, robust=False,
                    nuc='ple_Ex', mixer_deg=-90, eom_ampl =0.15,step_idx=0, laser_dur=10.0)
            mcas.asc(length_mus=1.0, nw = False)

            if _I_['state_check']:
                # Charge state check as EX + A1 readout
                sna.ssr(mcas, frequencies=freq, wait_dur=0.0, robust=False, nuc='charge_state', mixer_deg=-90,
                        step_idx=1, laser_dur=100.0)

            mcas.asc(length_mus=1.0)
            #mcas.asc(length_mus=10.0)

        pi3d.gated_counter.set_n_values(mcas)

        return mcas
    return ret_mcas

def settings(pdc={}):
    ana_seq=[
        ['result', '>', 1, 1, 0, 1],
        # ['init', '>', 1, 1, 0, 1],
        # ['init', '>', 5, 1, 0, 1],
    ]
    sch.settings(
        nuclear=nuclear,
        ret_mcas=ret_ret_mcas(pdc),
        analyze_sequence=ana_seq,
        pdc=pdc,
        meas_code=meas_code
    )

    nuclear.x_axis_title = 'tau_half [mus]'
    #nuclear.analyze_type = 'consecutive'
    nuclear.analyze_type = 'standard'

    nuclear.do_ple_refocusEx = True
    nuclear.do_ple_refocusA1 = False
    nuclear.do_ple_refocus = True
    nuclear.do_odmr_refocus = False
    nuclear.do_confocal_red_refocus = True

    nuclear.ple_refocus_interval = 180
    nuclear.confocal_red_refocus_interval = 180  # 240

    pi3d.gated_counter.trace.consecutive_valid_result_numbers = [0]
    pi3d.gated_counter.trace.average_results = False
    compress = 0.4 # 40% reduction of points
    tmax = 200 # us
    n_p = tmax*2
    indices = np.unique(np.geomspace(1,n_p,int(n_p*compress)).astype(int))-1
    nuclear.parameters = OrderedDict(
        (
            ('sweeps', range(100)),
            #('amp0',np.linspace(0.01,1.0,3)),
            #('amp0', [1.0]),
            ('rabi_period', [0.14]),
            ('resonant', [True]),
            ('ms', [-1]),
            ('state_check', [False]),
            ('nucl_init', [False]),
            ('additional_estate_check', [False]),
            #('ddt', ['fid', 'hahn', 'xy4', 'xy16', 'kdd','kdd4', 'kdd16']),
            ('ddt', ['hahn']),
            # ('mw_duration', [0.1]),
            ('aom_Ex_power_sweep', [1.0]),
            #('temp', [True]),
            ('n_rep_dd', [1]),
            ('delay_ps',[0]), #11110
            #('total_tau', np.hstack([[0.0], np.linspace(0.03, 50, 50)])),
            ('tau', [3.5]),
            #('tau', E.round_length_mus_full_sample(np.arange(3.0,50,2.0))),
            ('T_C1', E.round_length_mus_full_sample(np.linspace(0.0, tmax, n_p))[indices]),
            ('T_C2', E.round_length_mus_full_sample(np.linspace(0.1, tmax, n_p))[indices]),
            ('phase_pi2_2', [0.0, np.pi]),

        )
    )
    nuclear.number_of_simultaneous_measurements = len(nuclear.parameters['phase_pi2_2'])

def run_fun(abort, **kwargs):

    # pi3d.gated_counter.readout_duration = 5e6
    pi3d.gated_counter.readout_duration = 1e6*10

    nuclear.debug_mode = False
    settings()
    print('run_fun started')

    nuclear.run(abort)

