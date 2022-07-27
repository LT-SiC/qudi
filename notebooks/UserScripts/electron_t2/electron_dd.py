# coding=utf-8
#from pi3diamond import pi3d
import datetime
import numpy as np
import os
import importlib
import notebooks.UserScripts.helpers.sequence_creation_helpers as sch; importlib.reload(sch)
import notebooks.UserScripts.helpers.shared as shared
#import hardware.multi_channel_awg_seq as MCAS
#import notebooks.UserScripts.helpers.snippets_awg as sna

#importlib.reload(sna)
importlib.reload(shared)
#importlib.reload(MCAS)
import notebooks.UserScripts.helpers.shared as ush;importlib.reload(ush)
from logic.qudip_enhanced import *
#import hardware.Keysight_AWG_M8190.elements as e
from collections import OrderedDict


seq_name = os.path.basename(__file__).split('.')[0]
nuclear = sch.create_nuclear(__file__)
with open(os.path.abspath(__file__).split('.')[0] + ".py", 'r') as f:
    meas_code = f.read()

__TAU_HALF__ = 2*192/12e3
__SAMPLE_FREQUENCY__ = 12e3#e.__SAMPLE_FREQUENCY__

ael = 1.0

def ret_ret_mcas(pdc):
    def ret_mcas(current_iterator_df):
        sequence_name = 'Electron_t2_red_Tst'

        #mcas = MCAS.MultiChSeq(name=sequence_name, ch_dict={'2g': [1, 2], 'ps': [1]})
        #mcas.start_new_segment('start_sequence')
        #mcas.asc(length_mus=0.1)  # Starting... histogram 0




        #def erabi(freq, length, amp, phase=0.0):
        #    sna.electron_rabi(mcas,
        #                      name='electron rabi',
        #                      length_mus=length,
         #                     amplitudes=[amp],
        #                      frequencies=freq,
        #                      phases=np.rad2deg(phase),
        #                      new_segment=False,
        #                      mixer_deg=-90,
        #                      )

        #def waveform(seq):
         #   mcas.start_new_segment('waveform')
         #   sna.polarize(mcas, new_segment=False)
         #   mw = seq.times_fields_aphi('mw')
        #    wait = seq.times_fields_aphi('wait')
         #   for step in seq.sequence_steps:
         #       idx = int(step[1]) - 1
         #       if step[0] == 'mw':
         #           erabi(freq=freq, length=mw[idx, 0],
         #                 amp=pi3d.tt.rp('e_rabi', period=1 / mw[idx, 1], mixer_deg=-90).amp, phase=mw[idx, 2])
         #       elif step[0] == 'wait':
          #          mcas.asc(length_mus=wait[idx, 0])




        for idx, _I_ in current_iterator_df.iterrows():
            print(_I_)
            mcas = []

        #pi3d.gated_counter.set_n_values(mcas)

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
    #nuclear.analyze_type = 'standard'

    nuclear.do_ple_refocusEx = True
    nuclear.do_ple_refocusA1 = False
    nuclear.do_ple_refocus = True
    nuclear.do_odmr_refocus = False
    nuclear.do_confocal_red_refocus = True

    nuclear.ple_refocus_interval = 100
    nuclear.confocal_red_refocus_interval = 100  # 240

    #pi3d.gated_counter.trace.consecutive_valid_result_numbers = [0]
    #pi3d.gated_counter.trace.average_results = False

    nuclear.parameters = OrderedDict(
        (
            ('sweeps', range(100)),
            ('rabi_period', [0.07]),
            ('resonant', [True]),
            ('ms', [-1]),
            ('state_check', [False]),
            ('nucl_init', [False]),
            ('additional_estate_check', [False]),
            #('ddt', ['fid', 'hahn', 'xy4', 'xy16', 'kdd','kdd4', 'kdd16']),
            ('ddt', ['xy4']),
            ('n_rep_dd', range(1)),
            ('delay_ps',[0]), #11110
            ('tau', [1.0]),#E.round_length_mus_full_sample(np.linspace(0.0, 10.0, 100))),
            ('phase_pi2_2', [np.pi*0.5]),

        )
    )
    #nuclear.number_of_simultaneous_measurements =  1# len(nuclear.parameters['phase_pi2_2'])

def run_fun(abort, **kwargs):
    #pi3d.readout_duration = 1e6*100
    # pi3d.gated_counter.readout_duration = 5e6
    #pi3d.gated_counter.readout_duration = 1e6*10
    print(1,' Nuclear started!!!')
    nuclear.queue = kwargs['queue']
    nuclear.debug_mode = False
    settings()
    print('run_fun started')
    nuclear.run(abort)