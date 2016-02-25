#!/usr/bin/env python

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import pstats

params = {'backend': 'ps',
          'axes.labelsize': 10,
          'font.size': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,}
matplotlib.rcParams.update(params)


#Some simple functions to generate colours.
def pastel(colour, weight=2.4):
    """ Convert colour into a nice pastel shade"""
    rgb = np.asarray(colorConverter.to_rgb(colour))
    # scale colour
    #maxc = max(rgb)
    #if maxc < 1.0 and maxc > 0:
    #    # scale colour
    #    scale = 1.0 / maxc
    #    rgb = rgb * scale
    # now decrease saturation
    total = sum(rgb)
    slack = 0
    for x in rgb:
        slack += 1.0 - x

    # want to increase weight from total to weight
    # pick x s.t.  slack * x == weight - total
    # x = (weight - total) / slack
    x = (weight - total) / slack

    rgb = [c + 0.75*(x * (1.0-c)) for c in rgb]

    return rgb

def get_colours(n):
    """ Return n pastel colours. """
    base = np.asarray([[0.8,0.8,0], [0.8,0,0.8], [0,0.8,0.8]])

    if n <= 3:
        return base[0:n]

    # how many new colours do we need to insert between
    # red and green and between green and blue?
    needed = (((n - 3) + 1) / 2, (n - 3) / 2)
    
    colours = []
    for start in (0, 1):
        for x in np.linspace(0, 1-(1.0/(needed[start]+1)), needed[start]+1):
            colours.append((base[start] * (1.0 - x)) +
                           (base[start+1] * x))
    colours.append([0,0,1])

    return [pastel(c) for c in colours[0:n]]


time_components = {
                'CFL reduce' : "<method 'max' of 'petsc4py.PETSc.Vec' objects>",
                'Parallel initialization' : "<method 'create' of 'petsc4py.PETSc.DMDA' objects>",
                'Ghost cell communication' : "<method 'globalToLocal' of 'petsc4py.PETSc.DM' objects>",
                'Time evolution' : "evolve_to_time",
                'setup' : "setup",
                'Idle time' : "<method 'barrier' of 'petsc4py.PETSc.Comm' objects>"}


def extract_profile(ndim,solver_type,nvals,process_rank,ncells,base_dir):
    from scipy.special import cbrt
    import math
    stats_dir = base_dir + '/scaling_'+str(ndim)+'d_'+str(solver_type)
    
    times = {}
    for key in time_components.iterkeys():
        times[key] = []
    times['Python dynamic loading'] = []    
    times['Concurrent computations'] = []

    nprocs = []
    ngrids = []
    calc_ncells = (ncells==-1)

    for n in nvals:
        nprocs.append(n)
        if calc_ncells:
            ncells = int(96 * cbrt(n/4))
        ngrids.append(math.pow(ncells, ndim))
        prof_filename = os.path.join(stats_dir,'statst_'+str(n)+'_'+str(ncells)+'_'+str(process_rank))
        profile = pstats.Stats(prof_filename)
        prof = {}
        
        for key, value in profile.stats.iteritems():
            method = key[2]
            cumulative_time = value[3]
            prof[method] = cumulative_time
        for component, method in time_components.iteritems():
            times[component].append(round(prof[method],1))

        times['Python dynamic loading'].append(extract_results(ncells,ndim,solver_type,n,process_rank,base_dir))

    times['Concurrent computations'] =  [  times['Time evolution'][i]
                                        - times['CFL reduce'][i]
                                        - times['Ghost cell communication'][i]
                                        - times['Idle time'][i]
                                        for i in range(len(times['Time evolution']))]

    return nprocs,ngrids,times

def extract_results(ncells,ndim,solver_type,nval,process_rank,base_dir):
    import re
    #Extract dynamic python time
    logs_dir = base_dir + '/scaling_'+str(ndim)+'d_'+str(solver_type)
    log_filename = os.path.join(logs_dir,'results_'+str(nval)+'_'+str(ncells) + '.log')
    pattern = re.compile("Python dynamic loading: (\d+\.\d{2})")
    with open(log_filename, 'r') as log_filename_content:
        log_filename_content = log_filename_content.read()

    match = re.finditer(pattern, log_filename_content)
    for match in re.finditer(pattern, log_filename_content):
        return float(match.group(1))

    return 0

def plot_and_table(ndim=3,solver_type='classic',nvals=(16,),process_rank=0,ncells=-1,log_scale=False,base_dir='weak-scaling',show_gridsize=False):
    nprocs, ngrids, times = extract_profile(ndim,solver_type,nvals,process_rank,ncells,base_dir)

    rows = ['Concurrent computations',
            'Parallel initialization',
            'Ghost cell communication',
            'CFL reduce',
            'Idle time',
            'Python dynamic loading']

    # Get some pastel shades for the colours
    colours = get_colours(len(rows))
    nrows = len(rows)

    x_bar = np.arange(len(nprocs)) + 0.3  # the x locations for the groups
    bar_width = 0.4
    yoff = np.array([0.0] * len(nprocs)) # the bottom values for stacked bar chart

    plt.axes([0.35, 0.31, 0.55, 0.35])   # leave room below the axes for the table

    for irow,row in enumerate(rows):
        #print row + ': ' + str(times[row]) + '\n'
        plt.bar(x_bar, times[row], bar_width, bottom=yoff, color=colours[irow], linewidth=0,log=log_scale)
        yoff = yoff + times[row]

    table_data = [times[row] for row in rows]

    # Add total time to the table_data
    totol_times = np.array([sum([row[i] for row in table_data]) for i in range(len(nprocs))])
    table_data.append(totol_times)
    #table_data[-1] = table_data[-1][0]/table_data[-1]
    #table_data[-1] = [round(x,2) for x in table_data[-1]]
    rows.append('Total time')
    colours.append([1,1,1])

    # Add total efficiency to the table_data excluding the last two rows (python dynamic loading & total time)
    table_data.append(np.array([sum([row[i] for row in table_data[:-2]]) for i in range(len(nprocs))]))
    table_data[-1] = table_data[-1][0]/table_data[-1]
    table_data[-1] = [(table_data[-1][i]/(int(nprocs[i])/int(nprocs[0]))) * ngrids[i]/ngrids[0] for i in range(len(nprocs))]
    table_data[-1] = [round(x,2) for x in table_data[-1]]
    rows.append('Parallel efficiency')
    colours.append([1,1,1])

    if show_gridsize:
        # Add grid size
        table_data.append((np.power(ngrids, 1.0/ndim)/np.power(np.array([proc/nprocs[0] for proc in nprocs]), 1.0/ndim)))
        rows.append('Grid size (cubic root)')
        colours.append([1,1,1])

    # Add a table at the bottom of the axes
    mytable = plt.table(cellText=table_data,
                        rowLabels=rows, rowColours=colours,
                        colLabels=nprocs,
                        loc='bottom').set_fontsize(8)

    plt.ylabel('Execution Time for Process '+ str(process_rank)+' (s)')
    plt.figtext(.4, .01, "Number of Cores", fontsize=8)

    if not log_scale:
        ytickstep = 50
        yticksmax = int(((np.max(totol_times) // ytickstep) + 1) * ytickstep) + 1
        vals = np.arange(0, yticksmax + 1, ytickstep)
        plt.yticks(vals, ['%d' % val for val in vals])

    plt.xticks([])
    plt.draw()

    f=plt.gcf()
    f.set_figheight(7) 
    f.set_figwidth(5) 

    plt.savefig(base_dir + '/scaling_'+solver_type+'_'+str(ndim)+'D_' + str(process_rank) +'.pdf')
    plt.close()

def plot_concurrent_times(ndim=2,solver_type='classic',nvals=(64,),process_ranks=range(64),ncells=-1,base_dir='.'):
    x=[]
    y=[]

    for pr in process_ranks:
        nprocs, times = extract_profile(ndim,solver_type,nvals,pr,ncells,base_dir)
        x.append(pr)
        y.append(times['Concurrent computations'])

    plt.plot(x,y)

    plt.xlabel('process rank')
    plt.ylabel('time (s)')
    plt.grid(True)
    plt.savefig('Concurrent_'+solver_type+'_'+str(ndim)+'D_.pdf')
    plt.show()
    plt.close()

if __name__ == '__main__':
    plot_and_table()
