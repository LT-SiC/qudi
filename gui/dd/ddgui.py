# -*- coding: utf-8 -*-

"""
This file contains a gui for the DD.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

import numpy as np
import os
import pyqtgraph as pg
import time

from core.connector import Connector
from gui.colordefs import QudiPalettePale as palette
from gui.colordefs import ColorScaleInferno
from gui.guiutils import ColorBar
from gui.guibase import GUIBase
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy import uic

from gui.dd.connectors_and_set_default import initialize_connections_and_defaultvalue as dd_default_functions


class DDWindow(QtWidgets.QMainWindow):
    """ Create the Main Window based on the *.ui file. """

    def __init__(self):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_dd_gui.ui')

        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)
        self.show()

class DDGUI(GUIBase,dd_default_functions):

    ## declare connectors
    ddlogic = Connector(interface='DDLogic')

    # sigA1 = QtCore.Signal(bool)
    # sigA2 = QtCore.Signal(bool)
    # sigRepump = QtCore.Signal(bool)
    # sigGreen = QtCore.Signal(bool)
    # sigMW1ON = QtCore.Signal(bool)
    # sigMW2ON = QtCore.Signal(bool)
    # sigMW3ON = QtCore.Signal(bool)
    # sigSetPower = QtCore.Signal(bool)
    # sigPDzero = QtCore.Signal(bool)
    # sigAutofocus = QtCore.Signal(bool)
    # sigFlipMirror = QtCore.Signal(bool)


    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        
    
    def on_activate(self):
        """ Definition and initialisation of the GUI plus staring the measurement.
        """

        #self._setupcontrol_logic = self.setupcontrollogic() 
        #####################
        # Configuring the dock widgets
        # Use the inherited class 'CounterMainWindow' to create the GUI window
        self._mw = DDWindow()
        self._dd_logic= self.ddlogic()
        self.initialize_connections_and_defaultvalues()
        #outsource all the connectors into a second file, to keep the GUI file clean


        #connect the signals:
        self._dd_logic.sigDDPlotsUpdated.connect(self.update_plots, QtCore.Qt.QueuedConnection)
        self._dd_logic.SigClock.connect(self.Update_Runtime, QtCore.Qt.QueuedConnection)
        
        #get the number of points the Timetagger counter will return:
        self.number_of_points_per_line=self._dd_logic._time_tagger._counter["n_values"]

        #################### setup graphics for dd
        self.dd_matrix_image = pg.ImageItem(
                    np.random.rand(40, 20),
                    axisOrder='row-major')

        self.dd_matrix_image.setRect(QtCore.QRectF(
            #self._dd_logic.mw_starts[0],
            70,
            0,
            #self._dd_logic.mw_stops[0] - self._dd_logic.mw_starts[0],
            40,
            #self._dd_logic.number_of_lines
            20
        ))
        self.dd_data_image = pg.PlotDataItem(
                                        np.arange(20),
                                        np.arange(20),
                                        pen=pg.mkPen(palette.c3, style=QtCore.Qt.DotLine),
                                        symbol='o',
                                        symbolPen=palette.c3,
                                        symbolBrush=palette.c3,
                                        symbolSize=7)
        
        self.dd_data_image_2 = pg.PlotDataItem(
                                        np.arange(20),
                                        np.arange(20),
                                        pen=pg.mkPen(palette.c1, style=QtCore.Qt.DotLine),
                                        symbol='o',
                                        symbolPen=palette.c1,
                                        symbolBrush=palette.c1,
                                        symbolSize=7)
        
        self.dd_data_image_3 = pg.PlotDataItem(
                                        np.arange(20),
                                        np.arange(20),
                                        pen=pg.mkPen(palette.c5, style=QtCore.Qt.DotLine),
                                        symbol='o',
                                        symbolPen=palette.c5,
                                        symbolBrush=palette.c5,
                                        symbolSize=7)

        self.dd_data_image_fit = pg.PlotDataItem(
                                            np.arange(20),
                                            np.arange(20),
                                            pen=pg.mkPen(palette.c2))

        self.dd_detect_image = pg.PlotDataItem(
                                        np.arange(20),
                                        np.arange(20),
                                        pen=pg.mkPen(pg.mkColor(255, 255, 255), style=QtCore.Qt.DotLine),
                                        symbol='o',
                                        symbolPen=pg.mkColor(255, 255, 255),
                                        symbolBrush=pg.mkColor(255, 255, 255),
                                        symbolSize=7)

        self.dd_detect_fit_image = pg.PlotDataItem(
                                            np.arange(20),
                                            np.arange(20),
                                            pen=pg.mkPen(palette.c2))

        # Add the display item to the xy and xz ViewWidget, which was defined in the UI file.
        self._mw.dd_data_PlotWidget.addItem(self.dd_data_image)
        self._mw.dd_data_PlotWidget.addItem(self.dd_data_image_2)
        self._mw.dd_data_PlotWidget.addItem(self.dd_data_image_3)

        self._mw.dd_data_PlotWidget.addItem(self.dd_data_image_fit)
        self._mw.dd_data_PlotWidget.setLabel(axis='left', text='Counts', units='Counts')
        self._mw.dd_data_PlotWidget.setLabel(axis='bottom', text='Tau', units='s')
        self._mw.dd_data_PlotWidget.showGrid(x=True, y=True, alpha=0.8)

        self._mw.dd_detect_PlotWidget.addItem(self.dd_detect_image)
        self._mw.dd_detect_PlotWidget.setLabel(axis='left', text='Counts', units='Counts')
        self._mw.dd_detect_PlotWidget.setLabel(axis='bottom', text='Time', units='s')
        self._mw.dd_detect_PlotWidget.showGrid(x=True, y=True, alpha=0.8)

        self._mw.dd_matrix_PlotWidget.addItem(self.dd_matrix_image)
        self._mw.dd_matrix_PlotWidget.setLabel(axis='left', text='Tau', units='s')
        self._mw.dd_matrix_PlotWidget.setLabel(axis='bottom', text='Time', units='s')

        # Get the colorscales at set LUT
        my_colors = ColorScaleInferno()
        self.dd_matrix_image.setLookupTable(my_colors.lut)

        # Configuration of the Colorbar
        self.scan_cb = ColorBar(my_colors.cmap_normed, 100, 0, 100000)

        #adding colorbar to ViewWidget
        self._mw.dd_cb_PlotWidget.addItem(self.scan_cb)
        self._mw.dd_cb_PlotWidget.hideAxis('bottom')
        self._mw.dd_cb_PlotWidget.hideAxis('left')
        self._mw.dd_cb_PlotWidget.setLabel('right', 'Counts', units='c')

        # Connect the buttons and inputs for colorbar
        self._mw.dd_cb_manual_RadioButton.clicked.connect(self.update_colorbar)
        self._mw.dd_cb_centiles_RadioButton.clicked.connect(self.update_colorbar)
        self._mw.dd_cb_low_percentile_DoubleSpinBox.valueChanged.connect(self.update_colorbar)
        self._mw.dd_cb_high_percentile_DoubleSpinBox.valueChanged.connect(self.update_colorbar)
        self._mw.dd_cb_min_DoubleSpinBox.valueChanged.connect(self.update_colorbar)
        self._mw.dd_cb_max_DoubleSpinBox.valueChanged.connect(self.update_colorbar)
        self.cb_min = self._dd_logic.dd_cb_low_percentile
        self.cb_max = self._dd_logic.dd_cb_high_percentile

    def on_deactivate(self):
        """ Deactivate the module properly.
        """
        self.disconnect_all()
        
        self._dd_logic.sigDDPlotsUpdated.disconnect()
        self._dd_logic.SigClock.disconnect()
        self._mw.close()

    def show(self):
        """Make window visible and put it above all other windows.
        """
        QtWidgets.QMainWindow.show(self._mw)
        self._mw.activateWindow()
        self._mw.raise_()
    
    def restoreDefaultView(self):
        """ Restore the arrangement of DockWidgets to the default
        """
        # Show any hidden dock widgets
        self._mw.adjustDockWidget.show()
        self._mw.plotDockWidget.show()

        # re-dock any floating dock widgets
        self._mw.adjustDockWidget.setFloating(False)
        self._mw.plotDockWidget.setFloating(False)

        # Arrange docks widgets
        self._mw.addDockWidget(QtCore.Qt.DockWidgetArea(1), self._mw.adjustDockWidget)
        self._mw.addDockWidget(QtCore.Qt.DockWidgetArea(2), self._mw.plotDockWidget)

    def update_colorbar(self):
        if self._mw.dd_cb_centiles_RadioButton.isChecked():
            low_centile = self._mw.dd_cb_low_percentile_DoubleSpinBox.value()
            high_centile = self._mw.dd_cb_high_percentile_DoubleSpinBox.value()

            self.cb_min = np.percentile(self._dd_logic.scanmatrix, low_centile)
            self.cb_max = np.percentile(self._dd_logic.scanmatrix, high_centile)
        else:
            self.cb_min = self._mw.dd_cb_min_DoubleSpinBox.value()
            self.cb_max = self._mw.dd_cb_max_DoubleSpinBox.value()

        self.scan_cb.refresh_colorbar(self.cb_min, self.cb_max)
        self._mw.dd_cb_PlotWidget.update()
        self.dd_matrix_image.setImage(
                image=self._dd_logic.scanmatrix,
                axisOrder='row-major',
                levels=(self.cb_min, self.cb_max)
            )



    def update_plots(self):
        if self._dd_logic.measurement_running or self._dd_logic.update_after_stop:
            #print(dd_data_x, dd_data_y, dd_matrix)
            dd_data_x=np.array(self._dd_logic.sweep*1e-9, dtype=float)
            dd_data_y=np.array(self._dd_logic.data, dtype=float)
            dd_matrix=np.array(self._dd_logic.scanmatrix, dtype=float)
            dd_detect_x=np.array(self._dd_logic.measured_times, dtype=float)
            dd_detect_y=np.array(self._dd_logic.data_detect, dtype=float)
            self.dd_data_image.setData(dd_data_x[:len(dd_data_y)], dd_data_y[::2])# UNFUG
            self.dd_data_image_2.setData(dd_data_x[:len(dd_data_y)], dd_data_y[1::2])# UNFUG
            self.dd_data_image_3.setData(dd_data_x[:len(dd_data_y)], dd_data_y[::2]-dd_data_y[1::2])# UNFUG


            #self.dd_data_image.setData(dd_data_x, dd_data_y)

            self.dd_detect_image.setData(dd_detect_x, dd_detect_y)
            self._mw.dd_data_PlotWidget.removeItem(self.dd_data_image_fit)
            self._dd_logic.update_after_stop=False

            if self._dd_logic.dd_PerformFit:
                self._dd_logic.do_fit(self._dd_logic.sweep,self._dd_logic.data[::2],self._dd_logic.dd_FitFunction)
                self._mw.dd_FitResults_TextBrowser.setText(self._dd_logic.dd_FitParams)
                self._mw.dd_data_PlotWidget.addItem(self.dd_data_image_fit)
                self.dd_data_image_fit.setData(np.array(self._dd_logic.interpolated_x_data/1e9, dtype=float),np.array(self._dd_logic.fit_data, dtype=float))

            self.dd_matrix_image.setRect(
                QtCore.QRectF(
                    dd_detect_x[0],
                    dd_data_x[0],
                    dd_detect_x[-1]-dd_detect_x[0],
                    dd_data_x[-1]-dd_data_x[0]
                    )
            )

            self.dd_matrix_image.setImage(
                image=dd_matrix,
                axisOrder='row-major',
                levels=(self.cb_min, self.cb_max)
            )
            self.update_colorbar() # update the colorbar

    def Toggle_units_CheckBox_StateChanged(self,on):
        if on==2:#Amp
            self._mw.label_121.setText("Power (Amp)")
            self._mw.label_124.setText("Power (Amp)")
            self._mw.label_132.setText("Power (Amp)")
            self._mw.label.setText("Powers (Amp)")
        else:
            self._mw.label_121.setText("Power (dBm)")
            self._mw.label_124.setText("Power (dBm)")
            self._mw.label_132.setText("Power (dBm)")
            self._mw.label.setText("Powers (dBm)") 

        
        self._mw.dd_MW1_Power_LineEdit.setText(str(self._dd_logic.dd_MW1_Power))
        self._mw.dd_MW2_Power_LineEdit.setText(str(self._dd_logic.dd_MW2_Power))
        self._mw.dd_MW3_Power_LineEdit.setText(str(self._dd_logic.dd_MW3_Power))
        self._mw.dd_MW4_Power_LineEdit.setText(str(self._dd_logic.dd_MW4_Power))
        self._mw.dd_MW5_Power_LineEdit.setText(str(self._dd_logic.dd_MW5_Power))

    def Update_Runtime(self):
        if self._dd_logic.measurement_running:
            runtime=time.time()-self._dd_logic.starting_time
            hours=int(runtime//3600)
            minutes=int((runtime//60)%60)
            secs=int(runtime%60)
            #time_str=str(runtime).split(".")
            self._mw.dd_Runtime_Label.setText(f"{hours} h {minutes} m {secs} s")
