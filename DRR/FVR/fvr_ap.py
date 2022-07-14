# importing package
#import matplotlib.pyplot as plt
#import numpy as np
  
# create data
#x = np.arange(3)
#y1 = [7.045, 7.045, 7.045]
#y2 = [0.150, 0.719, 2.305]
#width = 0.40
  
# plot data in grouped manner of bar type
#plt.bar(x-0.2, y1, width)
#plt.bar(x+0.2, y2, width)
#plt.savefig("fvr_ap.png")

import matplotlib.pyplot as plt
import numpy as np


labels = [r'$400\times400$', r'$800\times800$', r'$1200\times1200$']
y1 = [7.05, 7.05, 7.05]
y2 = [0.15, 0.72, 2.31]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, y1, width, label='AP')
rects2 = ax.bar(x + width/2, y2, width, label='FVR')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Projection time (seconds)')
ax.set_xlabel('Image size')
#ax.set_title('Average projection time per image for the AP and FVR methods')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

#plt.show()
plt.savefig("fvr_ap_time.png")
