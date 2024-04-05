import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.transforms as transforms


mpl.rcParams['font.family'] = 'Source Han Sans SC'

stroke_white = [pe.withStroke(linewidth=2, foreground='white')]
stroke_black = [pe.withStroke(linewidth=2, foreground='black')]

# matplotlib 
FIG_PRESET = {
    'figure.dpi':163.18/2,
    'axes.linewidth':1,
    'grid.color':'lightgray',
}

@mpl.rc_context(FIG_PRESET)
def set_square_grid_fig(
        max_pull,          
        x_grids=10,         
        y_base_gap=50,      
        y2x_base=4/3,       
        y_force_gap=None,   
    ):

    graph_x_space = 5       
    x_pad = 1.2             
    y_pad = 1.2             
    title_y_space = 0.1     

    x_gap = 1 / x_grids     
    y_gap = y_base_gap * math.ceil((max_pull / ((x_grids+1) * max(y2x_base, math.log(max_pull)/5) - 1) / y_base_gap))
    if y_force_gap is not None:
        y_gap = y_force_gap

    y_grids = math.ceil(max_pull / y_gap)
    graph_y_space = (y_grids + 1) * graph_x_space / (x_grids + 1)  

    x_size = graph_x_space+x_pad
    y_size = graph_y_space+title_y_space+y_pad
    fig_size = [x_size, y_size]
    fig = plt.figure(figsize=fig_size) 
    
    ax = fig.add_axes([0.7*x_pad/x_size, 0.6*y_pad/y_size, graph_x_space/x_size, graph_y_space/y_size])
    ax.set_xticks(np.arange(0, 1.01, x_gap))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_yticks(np.arange(0, (y_grids+2)*y_gap, y_gap))

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.4*y_gap, (y_grids+0.4)*y_gap)

    ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
    ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    return fig, ax, x_gap, x_grids, y_gap, y_grids

def add_stroke_dot(ax, x, y, path_effects=stroke_white, zorder=10, *args, **kwargs):
    ax.scatter(
        x, y, zorder=zorder,
        path_effects=path_effects,
        *args, **kwargs
    ) 
    return ax


def default_item_num_mark(x):
    return str(x)+'个'


def get_default_description(
        item_name='tool',
        cost_name='wish',
        text_head=None,
        mark_exp=None,
        direct_exchange=None,
        show_max_pull=None,
        is_finite=None,
        text_tail=None
    ):
    description_text = ''
   
    if text_head is not None:
        description_text += text_head
 
    if mark_exp is not None:
        if description_text != '':
            description_text += '\n'
        description_text += 'invest all'+item_name+'wishes'+format(mark_exp, '.2f')+cost_name
        if direct_exchange is not None:
            description_text += '\nadd'+str(direct_exchange)+cost_name+'and'+item_name+'\nper character drawn'+format(1/(1/mark_exp+1/direct_exchange), '.2f')+cost_name

    if show_max_pull is not None:
        if is_finite is None:
            pass
        elif is_finite:
            if direct_exchange is None:
                if description_text != '':
                    description_text += '\n'
                description_text += 'invest'+item_name+'items'+str(show_max_pull)+cost_name
        else:
            if description_text != '':
                description_text += '\n'
            description_text += 'Drawn per basic distrubition for '+item_name
  
    if text_tail is not None:
        if description_text != '':
            description_text += '\n'
        description_text += text_tail
    description_text =  description_text.rstrip()
    return description_text

# Add quantile function
def add_quantile_line(
        ax,
        data: np.array,
        color='C0',
        linewidth=2.5,
        max_pull=None,
        add_end_mark=False,
        is_finite=True,
        quantile_pos: list=None,
        item_num=1,
        add_vertical_line=False,
        y_gap=50,
        mark_pos=0.5,
        text_bias_x=-3/100,
        text_bias_y=3/100,
        mark_offset=-0.3,
        path_effects=stroke_white,
        mark_func=default_item_num_mark,
        *args, **kwargs
    ):
    
    if max_pull is None:
        max_pull = len(data)
  
    ax.plot(data[:max_pull],
            range(max_pull),
            linewidth=linewidth,
            color=color,
            *args, **kwargs
            )
    
    if quantile_pos is not None:
        offset_1 = transforms.ScaledTranslation(text_bias_x, text_bias_y, plt.gcf().dpi_scale_trans)
        offset_2 = transforms.ScaledTranslation(mark_offset+text_bias_x, text_bias_y, plt.gcf().dpi_scale_trans)
        offset_3 = transforms.ScaledTranslation(1.3*text_bias_x, text_bias_y, plt.gcf().dpi_scale_trans)
        transform_1 = ax.transData + offset_1
        transform_2 = ax.transData + offset_2
        transform_3 = ax.transData + offset_3
       
        for p in quantile_pos:
            pos = np.searchsorted(data, p, side='left')
            if pos >= len(data):
                continue
           
            dot_y = (p-data[pos-1])/(data[pos]-data[pos-1])+pos-1
            add_stroke_dot(ax, p, dot_y, s=3, color=color)
           
            ax.text(p, dot_y, str(pos),
                    weight='medium',
                    size=12,
                    color='black',
                    transform=transform_1,
                    horizontalalignment='right',
                    path_effects=path_effects
                )
            
            if p == mark_pos:
                plt.text(p, dot_y, mark_func(item_num),
                    weight='bold',
                    size=12,
                    color=color,
                    transform=transform_2,
                    horizontalalignment='right',
                    path_effects=path_effects
                )
          
            if add_vertical_line:
                ax.plot([p, p], [-y_gap/2, dot_y+y_gap*2], c='gray', linewidth=2, linestyle=':')
                ax.text(p, dot_y+y_gap*1.5, str(int(p*100))+"%",
                    transform=transform_3,
                    color='gray',
                    weight='bold',
                    size=12,
                    horizontalalignment='right',
                    path_effects=path_effects
                )
    if add_end_mark:
       
        if is_finite:
            add_stroke_dot(ax, 1, len(data)-1, s=10, color=color, marker="o")
       
        else:
            offset = transforms.ScaledTranslation(0, 0.01, plt.gcf().dpi_scale_trans)
            transform = ax.transData + offset
            add_stroke_dot(ax, data[-1], max_pull, s=40, color=color, marker="^", transform=transform, path_effects=[])
    return ax

def plot_pmf(ax, input: np.array, line_color='C0', dist_end=True, is_step=True, fill_alpha=0.5, path_effects=[pe.withStroke(linewidth=3, foreground='white')]):
    x = np.arange(len(input))        
    y = input
    if is_step:
        # x = np.arange(len(input)) + 0.5 
        # x = np.repeat(x, 2)           
        # y = np.repeat(input, 2)             

        # 
        # if input[0] != 0:
        #     x = np.append(-0.5, x[:-1]) 
        # else:
        #     x = x[1:-1]
        #     y = y[2:]
        step = 'mid'
    else:
        if input[0] == 0:
            x = x[1:]
            y = y[1:]
        step = None
  
    ax.fill_between(x, 0, y, alpha=fill_alpha, color=line_color, zorder=9, step=step, edgecolor='none')
    if step is None:
        ax.plot(
            x, y,
            color=line_color,
            linewidth=1.5,
            path_effects=path_effects,
            zorder=10)
    else:
        ax.step(
            x, y,
            color=line_color,
            linewidth=1.5,
            path_effects=path_effects,
            zorder=10,
            where=step)

   
    if not dist_end:
        add_stroke_dot(ax, x[-1], y[-1], s=10, color=line_color, marker=">", path_effects=path_effects)
        ax.plot(x, y, color=line_color, linewidth=1.5, zorder=10)
    
    return ax

def plot_cdf(ax, input: np.array, line_color='C0', dist_end=True, is_step=True, fill_alpha=0.5, path_effects=[pe.withStroke(linewidth=3, foreground='white')]):
    if is_step:
        x = np.arange(len(input)) + 0.5 
        x = np.repeat(x, 2)            
        y = np.repeat(input, 2)        

        
        if input[0] != 0:
            x = np.append(-0.5, x[:-1])  
        else:
            x = x[1:-1]
            y = y[2:]
    else:
        x = np.arange(len(input))         
        y = input
        if input[0] == 0:
            x = x[1:]
            y = y[1:]
    # cdf
    ax.plot(
        x, y,
        linewidth=1.5,
        color=line_color,
        zorder=10,
        path_effects=path_effects,
        )
    ax.fill_between(x, 0, y, alpha=fill_alpha, color=line_color, zorder=9)


    if not dist_end:
        add_stroke_dot(ax, x[-1], y[-1], s=10, color=line_color, marker=">", path_effects=path_effects)
        ax.plot(x, y, linewidth=1.5, color=line_color, zorder=10)
    return ax

def add_vertical_quantile_pmf(ax, pdf: np.ndarray, quantile_pos:list, mark_name='wishes', color='gray', pos_func=lambda x:x, pos_rate=1.1, size=10, show_mark_name=True):
   
    cdf = np.cumsum(pdf)
    x_start = ax.get_xlim()[0]
    y_start = ax.get_ylim()[0]
    offset_v = transforms.ScaledTranslation(0, 0.05, plt.gcf().dpi_scale_trans)
    for p in quantile_pos:
        pos = np.searchsorted(cdf, p, side='left')
        if pos >= len(cdf):
            continue
        # ax.plot([x_start, pos], [cdf[pos], cdf[pos]])
        ax.plot([pos, pos], [y_start, max(pdf)*pos_rate],
                color=color, zorder=2, alpha=0.75, linestyle=':', linewidth=2)
        
        if show_mark_name:
            show_text = str(pos_func(pos))+mark_name+'\n'+str(int(p*100))+"%"
        else:
            show_text = str(int(p*100))+"%"
        ax.text(pos, max(pdf)*pos_rate, show_text,
                horizontalalignment='center',
                verticalalignment='bottom',
                weight='bold',
                color='gray',
                size=size,
                transform=ax.transData + offset_v,
                path_effects=stroke_white,
                )

if __name__ == '__main__':
    #import games.genshin_impact as GI
    # fig, ax, _, _, y_grids, y_gap = set_square_grid_fig(max_pull=200)
    # # add_stroke_dot(ax, 0.5, 500, s=3)
    # add_vertical_cdf(
    #     ax,
    #     GI.up_5star_character(1).cdf,
    #     quantile_pos=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    #     add_vertical_line=True,
    #     is_finite=False,
    #     add_end_mark=True,
    # )
    # description_text = get_default_description(mark_exp=GI.up_5star_character(1).exp)
    # ax.text(
    #     0, y_grids*y_gap,
    #     description_text,
    #     weight='bold',
    #     size=12,
    #     color='#B0B0B0',
    #     path_effects=stroke_white,
    #     horizontalalignment='left',
    #     verticalalignment='top'
    # )
    
    # plt.show()