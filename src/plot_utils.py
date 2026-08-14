from pathlib import Path
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib


# Try to add arial font to the font manager
font_dirs = ['.']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
if len(font_files) == 0:
    print("No fonts found")
else:
    for font_file in font_files:
        print(font_file)
        font_manager.fontManager.addfont(font_file)

        if Path(font_file).name == "arial.ttf":
            print(Path(font_file))
            plt.rcParams['font.family'] = 'Arial'

matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 
matplotlib.rc('axes', labelsize=10)

plt.rcParams.update({
    'axes.titlesize': 10,   # Font size for axes titles
    'axes.labelsize': 10,   # Font size for axes labels
    'xtick.labelsize': 10,  # Font size for x-axis tick labels
    'ytick.labelsize': 10,  # Font size for y-axis tick labels
    'legend.fontsize': 10,  # Font size for legend
    'figure.titlesize': 10  # Font size for figure title
})

# Customising legend
# https://stackoverflow.com/questions/40672088/matplotlib-customize-the-legend-to-show-squares-instead-of-rectangles
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch

# --- handlers ---

class HandlerRect(HandlerPatch):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        x = width//2
        y = 0
        w = h = 10

        # create
        p = patches.Rectangle(xy=(x, y), width=w, height=h)

        # update with data from oryginal object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]

class HandlerCircle(HandlerPatch):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        r = 5
        x = r + width//2
        y = height//2

        # create 
        p = patches.Circle(xy=(x, y), radius=r)

        # update with data from oryginal object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]


rect = patches.Rectangle((0,0), 1, 1, facecolor='#FF605E')
circ = patches.Circle((0,0), 1, facecolor='#64B2DF')

leg_handler_map={
               patches.Rectangle: HandlerRect(),
               patches.Circle: HandlerCircle(),
            }
