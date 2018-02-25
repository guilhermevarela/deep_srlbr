import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_intervals(s, f, v, rfilter=None):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,aspect='equal')

    bars = []
    for i in range(len(s)):
        if rfilter is None or rfilter[i]:
            r = patches.Rectangle((s[i],i),f[i]-s[i],0.5,fill='b',alpha=v[i]/max(v))
            bars.append(r)

    for p in bars:
        ax.add_patch(p)
    plt.xlim((min(s),max(f)))
    plt.ylim((0,len(f)))
    plt.show()