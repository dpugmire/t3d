import re
from termcolor import colored


class logbook:
    '''
    Class to handle writing terminal output to the terminal AND a file.
    It uses the termcolor class to colorize terminal output. The color
    is removed for file output as many editors will not be able to show
    the colors. Using `less -R` or `cat` is okay but opening with an
    editor like vi or emacs shows ascii characters).

    The termcolor class provides the following color options: red,
    green, yellow, blue, magenta, and cyan as well as light_ versions
    '''

    def __init__(self):
        ''' Default initialization, set terminal stream to true and file to false '''
        self.term_stream = True   # Terminal output on by default
        self.file_stream = False  # File output off by default

    def __getstate__(self):
        ''' Only pickle stream variables (to avoid trying to pickle the output file) '''
        return {'term_stream': self.term_stream, 'file_stream': self.file_stream}

    def __setstate__(self, streams):
        ''' Unpickle stream variables '''
        self.term_stream = streams['term_stream']
        self.file_stream = streams['file_stream']

    def set_handlers(self,
                     term_stream=True,
                     file_stream=False,
                     file_handler='log.out'):
        ''' Initializes the logbook handlers as requested by the user '''
        self.term_stream = term_stream
        self.file_stream = file_stream
        if self.file_stream:
            self.file_handler = open(file_handler, 'w', buffering=1)

    def finalize(self):
        ''' Finalizes the logbook handlers, closes the file stream '''
        if self.file_stream:
            self.file_handler.close()

    def remove_color(self, text):
        ''' Function to strip colors from string '''
        no_colors = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return no_colors.sub('', text)

    def info(self, msg: str, color=None):
        ''' Output function with specified color '''
        if self.term_stream:
            print(colored(text=msg, color=color))
        if self.file_stream:
            self.file_handler.write(self.remove_color(msg) + '\n')

    def bold(self, msg: str, color='black'):
        ''' Output function with specified color and also bold '''
        if self.term_stream:
            print(colored(text=msg, color=color, attrs=['bold']))
        if self.file_stream:
            self.file_handler.write(self.remove_color(msg) + '\n')

    def emph(self, msg: str):
        ''' Output function in blue and bold for emphasis '''
        if self.term_stream:
            print(colored(text=msg, color='blue', attrs=['bold']))
        if self.file_stream:
            self.file_handler.write(self.remove_color(msg) + '\n')

    def warn(self, msg: str):
        ''' Output function in bold yellow with a warning tag '''
        if self.term_stream:
            print(colored(text='WARNING:' + msg, color='yellow', attrs=['bold']))
        if self.file_stream:
            self.file_handler.write('WARNING:' + self.remove_color(msg) + '\n')

    def errr(self, msg: str):
        ''' Output function in bold red with an error tag '''
        if self.term_stream:
            print(colored(text='ERROR:' + msg, color='red', attrs=['bold']))
        if self.file_stream:
            self.file_handler.write('ERROR:' + self.remove_color(msg) + '\n')

    def debug(self, msg: str):
        ''' Output function for debugging which only goes to the file stream '''
        if self.file_stream:
            self.file_handler.write(self.remove_color(msg) + '\n')


# Logbook instantiation for t3d
log = logbook()

# Short-cut functions
info = log.info
bold = log.bold
emph = log.emph
warn = log.warn
errr = log.errr
debug = log.debug
