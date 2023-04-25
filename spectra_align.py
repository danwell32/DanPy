#Coding for the align oxygen function
#
import hyperspy.api as hs
import hyperspy as hspy
import numpy as np 
from time import time

#External function defined to time the processes internally as a wrapper (decorator)

def timer(func):
    #
    '''
    Function defined as timer for other methods
    ________________________________________________________________
    It is thought to be used as a decorator @timer on top of methods
    '''
    def f(*args,**kwargs):
        ti = time()
        rv = func(*args,**kwargs)
        tf = time()
        print('Elapsed time: {} s'.format(tf-ti))
        return rv
    return f


class spectra_align():
    '''
    The idea of this class is to provide methods to allow interactive
    alignment of the spectrum or spectrum images desired.
    '''


    def __init__(self, s, workcopy=True, *args, **kw):
        '''
        Initialization method

        Prepares the spectra lo be align, respect the area we select in an 
        interactive ROI
        ###########################
        Parameters:

        s = hyperspy spectra ort spectrum image
        workcopy = bool(). Decides if we create a shallow copy of the spectra, 
                           to avoid changing it live. Default = True

        Class parameters created:

        self.si = hyperspy spectrum or spectrum image
        self.inav_pos = tuple().Position of reference in the SI
        self.multiple = bool(). If True, SI. Else, single S
        self.dicti_align = dict(). Dictionary to keep track of changes
        self.E_axis = np.array(). Array of energy values for the E axis

        '''

        self.si = s.deepcopy() if workcopy else s
        self.dicti_align = dict()
        self.E_axis = self.si.axes_manager[-1].axis
        self.dicti_align['initial_limits'] =\
        tuple(self.E_axis[int(i*len(self.E_axis)/3)] for i in [1,2])
        self.lefty = self.dicti_align['initial_limits'][0]
        self.righty = self.dicti_align['initial_limits'][1]


        ### Spectrum image or Single spectrum ###

        if self.si.axes_manager.indices != ():

            if len(self.si.axes_manager.indices) > 1:
                tuppy = tuple(int(ti/2) for ti in self.si.data.shape[:-1])
                self.inav_pos = (tuppy[-1],tuppy[0])
                self.si.axes_manager.indices = self.inav_pos
                self.multiple = True
                self.dicti_align['Type_spectra'] = 'Spectrum_Image'

            elif len(self.si.axes_manager.indices) == 1:
                tuppy = tuple(int(ti/2) for ti in self.si.data.shape[:-1])
                self.inav_pos = tuppy
                self.si.axes_manager.indices = self.inav_pos
                self.multiple = True
                self.dicti_align['Type_spectra'] = 'Spectrum_Line'



        else:
            self.multiple = False
            self.dicti_align['Type_spectra'] = 'Single_Spectra'

    def align_region_window(self,*args,**kwargs):
        '''
        Method to interactively change the window in which the maximum
        will be looked for for alignment

        RUN this and close the interactive window launched to change the
        limits of the search. If not, limits will not be taken into account

        Class parameters created:
        self.s_window = hs.spectra where the roi setup is performed
        self.roi = ROI element to correct alignment

        '''
        self.roi = hs.roi.SpanROI(left = self.lefty,right = self.righty)

        if self.multiple:
            self.s_window = self.si.inav[self.inav_pos].deepcopy()

        else:
            self.s_window = self.si.deepcopy()

        self.s_window.plot()
        self.roi.interactive(self.s_window,color = 'g')

    def alignment_correction(self,reference_element = 'Oxygen',reference_onset = 532.,*args,**kwargs):
        '''
        Method to aply the correction to the spectra, based on the area selected
        ############
        Parameters:
        reference_element = str(). Element that will serve as a reference for
                            alignment. Default = 'Oxygen'
        reference_onset = float(). Energy value for the theoretical E-Onset of
                          the reference_element. Dafualt = 532. Oxygen onset 

        Class parameters created:
        self.dicti_aling['Onset_displacement'] = value of the deviation of onset
                                                 respect to the theory
        self.dicti_align['final_limits'] = tuple(). Values of the interactive ROI
        self.dicti_align['reference_element'] = tuple(). Element and onset

        self.final_si = hs.spectra. The final spectra with the corrected axis


        '''
        self.final_si = self.si.deepcopy()
        self.dicti_align['reference_element'] = (reference_element,reference_onset)
        str00 = 'Check that the element selected (default = \'Oxigen\')'
        str01 = 'belongs to the acquired spectrum. If not, change element name and onset'
        str02 = 'in the function call (reference_element, reference_onset)' 

        print('\n'.join([str00,str01,str02]))

        self.dicti_align['final_limits'] = (self.roi.left, self.roi.right)

        s_interest = self.s_window.isig[self.roi.left:self.roi.right].deepcopy()
        array_axis = s_interest.axes_manager[-1].axis
        idx = abs(np.amax(s_interest.data)-s_interest.data).argmin()
        distance = reference_onset - array_axis[int(idx)]

        self.dicti_align['Onset_displacement'] = distance

        #We act on the offset of the initial spectrum. 

        offset0 = self.si.axes_manager[-1].trait_get('offset')['offset']

        self.final_si.axes_manager[-1].trait_set(offset = offset0 + distance)

        #Now, we return the spectra so it can stored in a new object outside

        return self.final_si








