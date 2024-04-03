from GGanalysis.plot_tools import *
from GGanalysis.distribution_1d import FiniteDist, pad_zero
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cm as cm
import os

__all__ = [
    'QuantileFunction',
    'DrawDistribution',
]

class QuantileFunction(object):
    def __init__(self,
                dist_data: list=None,           
                title='Function against draw number', 
                item_name='tool',               
                save_path='figure',             
                y_base_gap=50,                  
                y2x_base=4/3,                   
                y_force_gap=None,               
                is_finite=True,                 
                direct_exchange=None,           
                plot_direct_exchange=False,     
                max_pull=None,                  
                line_colors=None,               
                mark_func=default_item_num_mark,
                mark_offset=-0.3,               
                text_head=None,                 
                text_tail=None,                 
                mark_exp=True,                  
                mark_max_pull=True,             
                description_func=get_default_description,
                cost_name='抽'
                ) -> None:
        # input check
        if line_colors is not None and len(dist_data) != len(line_colors):
            raise ValueError("Item number must match colors!")
        
        # Frequently modified parameters
        self.title = title
        self.item_name = item_name
        self.save_path = save_path
        for i, data in enumerate(dist_data):  
            if isinstance(data, np.ndarray):
                dist_data[i] = FiniteDist(data)
        self.data = dist_data
        self.is_finite = is_finite
        self.y_base_gap = y_base_gap
        self.y2x_base = y2x_base
        self.y_force_gap = y_force_gap
        self.mark_func = mark_func
        self.mark_offset = mark_offset
        self.y_gap = self.y_base_gap
        self.x_grids = 10
        self.x_gap = 1 / self.x_grids
        self.text_head = text_head
        self.text_tail = text_tail
        self.mark_exp = mark_exp
        self.mark_max_pull = mark_max_pull
        self.direct_exchange = direct_exchange
        self.plot_direct_exchange = False
        if self.direct_exchange is not None:
            if plot_direct_exchange:
                self.is_finite = True
                self.plot_direct_exchange = True
        
        self.description_func = description_func
        self.cost_name = cost_name
        
        # Default value
        self.xlabel = 'Get probability'
        self.ylabel = f'invest{self.cost_name}number'
        self.quantile_pos = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        self.mark_pos = 0.5
        self.stroke_width = 2
        self.stroke_color = 'white'
        self.text_bias_x = -3/100 #-3/100
        self.text_bias_y = 3/100
        self.plot_path_effect = [pe.withStroke(linewidth=self.stroke_width, foreground=self.stroke_color)]
        
        if dist_data is not None:
            self.data_num = len(self.data)
            self.exp = self.data[1].exp  
        else:
            self.data_num = 0
            self.exp = None
        if max_pull is None:
            if dist_data is None:
                self.max_pull = 100
            else:
                self.max_pull = len(self.data[-1]) - 1
        else:
            self.max_pull = max_pull
        if line_colors is None:
            self.line_colors = cm.Blues(np.linspace(0.5, 0.9, self.data_num))
        else:
            self.line_colors = line_colors

        # cdf
        cdf_data = []
        for i, data in enumerate(self.data):
            if self.is_finite: 
                cdf_data.append(data.cdf)
            else:
                cdf_data.append(pad_zero(data.dist, self.max_pull)[:self.max_pull+1].cumsum())
        # Well drawing required
        if self.plot_direct_exchange:
            calc_cdf = []
            for i in range(len(self.data)):
                if i == 0:
                    calc_cdf.append(np.ones(1, dtype=float))
                    continue

                ans_cdf = np.copy(cdf_data[i][:i*self.direct_exchange+1])
                for j in range(1, i):
                    b_pos = self.direct_exchange*(i-j)
                    e_pos = self.direct_exchange*(i-j+1)
                    fill_ans = np.copy(cdf_data[j][b_pos:e_pos])
                    ans_cdf[b_pos:e_pos] = np.pad(fill_ans, (0, len(ans_cdf[b_pos:e_pos])-len(fill_ans)), 'constant', constant_values=1)
                if i*self.direct_exchange+1 <= len(ans_cdf):
                    ans_cdf[i*self.direct_exchange] = 1
                calc_cdf.append(ans_cdf)
            self.cdf_data = calc_cdf
        else:
            self.cdf_data = cdf_data

    # draw image
    def show_figure(self, dpi=300, savefig=False):
        fig, ax, _, _, y_grids, y_gap = set_square_grid_fig(
            max_pull=self.max_pull,
            x_grids=self.x_grids,
            y_base_gap=self.y_base_gap,
            y2x_base=self.y2x_base,
            y_force_gap=self.y_force_gap
        )
        fig.set_dpi(dpi)
        ax.set_title(self.title, weight='bold', size=18)
        ax.set_xlabel('figure', weight='medium', size=12)
        ax.set_ylabel(f'invest{self.cost_name}number', weight='medium', size=12)
        
        for i, (data, color) in enumerate(zip(self.cdf_data[1:], self.line_colors[1:])):
            add_quantile_line(
                ax,
                data[:self.max_pull+1],
                color=color,
                item_num=i+1,
                linewidth=2.5,
                quantile_pos=self.quantile_pos,
                add_vertical_line=(i+1==len(self.cdf_data)-1),
                is_finite=self.is_finite,
                add_end_mark=True,
                mark_func=self.mark_func,
            )

        description_text = self.description_func(
            item_name=self.item_name,
            cost_name=self.cost_name,
            text_head=self.text_head,
            mark_exp=self.exp,
            direct_exchange=self.direct_exchange,
            show_max_pull=len(self.data[1])-1,  
            is_finite=self.is_finite,
            text_tail=self.text_tail,
            )  

        # Add description
        ax.text(
            0, y_grids*y_gap,
            description_text,
            weight='bold',
            size=12,
            color='#B0B0B0',
            path_effects=stroke_white,
            horizontalalignment='left',
            verticalalignment='top'
        )

        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, self.title+'.png'), dpi=dpi)
        else:
            plt.show()

class DrawDistribution(object):
    def __init__(   self,
                    dist_data=None,  # dist 1d based
                    max_pull=None,
                    title='The number of draws required to obtain items',
                    item_name='tool',
                    cost_name='wish',
                    save_path='figure',
                    show_description=True,
                    quantile_pos=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
                    description_pos=0,
                    show_exp=True,
                    show_peak=True,
                    is_finite=True,
                    text_head=None,
                    text_tail=None,
                    description_func=get_default_description,
                ) -> None:
        # Initialization parameters
        if isinstance(dist_data, np.ndarray):
            dist_data = FiniteDist(dist_data)
        self.dist_data = dist_data
        self.max_pull = len(dist_data) if max_pull is None else max_pull
        self.show_description = show_description
        self.title = title
        self.item_name = item_name
        self.cost_name = cost_name
        self.text_head = text_head
        self.text_tail = text_tail
        self.description_pos = description_pos
        self.save_path = save_path
        self.description_func = description_func
        self.show_exp = show_exp
        self.show_peak = show_peak

        self.quantile_pos = quantile_pos
        self.is_finite = is_finite

        self.x_free_space = 1/40
        self.x_left_lim = 0-self.max_pull*self.x_free_space
        self.x_right_lim = self.max_pull+self.max_pull*self.x_free_space
        # ==============Distribution max================
        self.max_pos = dist_data[:].argmax()
        self.max_mass = dist_data.dist[self.max_pos]

        self.switch_len = 200
        self.plot_path_effect = stroke_white

    def show_dist(self, figsize=(9, 5), dpi=300, savefig=False, title=None):
        #figure
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(figsize)

        if title is None:
            ax.set_title(f"invest{self.cost_name}wishes", weight='bold', size=15)
            title = self.title
            fig.suptitle(self.title, weight='bold', size=20)
        else:
            ax.set_title(title, weight='bold', size=15)
   
        self.add_dist(ax, quantile_pos=self.quantile_pos)
 
        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, title+'.png'), dpi=dpi)
        else:
            plt.show()

    def show_cdf(self, figsize=(9, 5), dpi=300, savefig=False, title=None):
        # cdf
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(figsize)
        
        if title is None:
            ax.set_title("draw probability", weight='bold', size=15)
            title = self.title
            fig.suptitle(self.title, weight='bold', size=20)
        else:
            ax.set_title(title, weight='bold', size=15)
        
        self.add_cdf(ax, show_title=False, show_xlabel=True)
        
        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, title+'.png'), dpi=dpi)
        else:
            plt.show()

    def show_two_graph(
            self,
            savefig=False,
            figsize=(9, 8),
            dpi=300,
            color='C0',
            ):
 
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.set_size_inches(figsize)
        '''
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 3], figure=fig)
        axs[0].set_position(gs[0].get_position(fig))
        axs[1].set_position(gs[1].get_position(fig))
        '''
        ax_dist = axs[0]
        ax_cdf = axs[1]

        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, self.title+'.png'), dpi=dpi)
        else:
            plt.show()

    def add_dist(  
                    self, 
                    ax,
                    quantile_pos=None,
                    show_xlabel=True,
                    show_grid=True,
                    show_description=True,
                    fill_alpha=0.5,
                    minor_ticks=10,
                    main_color='C0',
                ):
        
        dist = self.dist_data
        # Set x/y range
        ax.set_xlim(self.x_left_lim, self.x_right_lim)
        ax.set_ylim(0, self.max_mass*1.26)
    
        if show_grid:
            ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
            ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
        
        # Switch the drawing mode according to the density
        if(len(dist) <= self.switch_len):
            plot_pmf(ax, dist[:self.max_pull], main_color, self.is_finite, is_step=True, fill_alpha=fill_alpha)
        else:
            plot_pmf(ax, dist[:self.max_pull], main_color, self.is_finite, is_step=False, fill_alpha=fill_alpha)
        
        exp_y = (int(dist.exp)+1-dist.exp) * dist.dist[int(dist.exp)] + (dist.exp-int(dist.exp)) * dist.dist[int(dist.exp+1)]
        # ax.axvline(x=dist.exp, c="lightgray", ls="--", lw=2, zorder=5, 
        #             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        
        if quantile_pos is not None:
            add_vertical_quantile_pmf(ax, dist[:self.max_pull], mark_name=self.cost_name, quantile_pos=self.quantile_pos)
            add_stroke_dot(ax, dist.exp, exp_y, color=main_color, s=10, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

        if self.show_peak:
            ax.text(
                self.max_pos, self.max_mass*1.01, 'figure'+str(self.max_pos)+self.cost_name,
                color='gray',
                weight='bold',
                size=10,
                horizontalalignment='center',
                verticalalignment='bottom',
                zorder=11,
                path_effects=self.plot_path_effect
            )
            
            add_stroke_dot(ax, self.max_pos, self.max_mass, color=main_color, s=10, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

        
        description_text = self.description_func(
            item_name=self.item_name,
            cost_name=self.cost_name,
            text_head=self.text_head,
            mark_exp=self.dist_data.exp,
            show_max_pull=len(dist)-1,  
            is_finite=self.is_finite,
            text_tail=self.text_tail,
            )  
        if show_description:
            ax.text(
                self.description_pos, self.max_mass*1.1,
                description_text,
                weight='bold',
                size=12,
                color='#B0B0B0',
                path_effects=self.plot_path_effect,
                horizontalalignment='left',
                verticalalignment='top',
                zorder=11)
    
    
    def add_cdf(    
                    self,
                    ax,
                    title='cumulative distribution function',
                    main_color='C0',
                    minor_ticks=10,
                    show_title=True,
                    show_grid=True,
                    show_xlabel=True,
                    show_description=False,
                ):
        dist = self.dist_data
        cdf = dist.dist.cumsum()

        # Set title and tags
        if show_title:
            ax.set_title(title, weight='bold', size=15)
        if show_xlabel:
            ax.set_xlabel(f"{self.cost_name}number", weight='bold', size=12, color='black')
        ax.set_ylabel('progressive probability', weight='bold', size=12, color='black')

        ax.set_xlim(self.x_left_lim, self.x_right_lim)
        ax.set_ylim(0,1.15)
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
      
        if show_grid:
            ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
            ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    
        if(len(dist) <= self.switch_len):          
            plot_cdf(ax, cdf[:self.max_pull], line_color=main_color, dist_end=self.is_finite, is_step=True)
        else:
            plot_cdf(ax, cdf[:self.max_pull], line_color=main_color, dist_end=self.is_finite, is_step=False)

        # ax.plot(range(1, len(dist)),cdf[1:],
        #         color=main_color,
        #         linewidth=3,
        #         path_effects=[pe.withStroke(linewidth=3, foreground='white')],
        #         zorder=10)
        # 
        # if self.is_finite is False:
        #     ax.scatter( len(cdf)-1, cdf[len(dist)-1],
        #                 s=10, color=main_color, marker=">", zorder=11)
        # 
        ax.grid(visible=True, which='major', color='lightgray', linestyle='-', linewidth=1)
        
        offset = transforms.ScaledTranslation(-0.05, 0.01, plt.gcf().dpi_scale_trans)
        trans = ax.transData + offset
        for p in self.quantile_pos:
            pos = np.searchsorted(cdf, p, side='left')
            if pos >= len(cdf): 
                continue
            # ax.plot([self.x_left_lim-1, pos], [cdf[pos], cdf[pos]], c="lightgray", linewidth=2, linestyle="--", zorder=0)
            ax.plot([pos, pos], [-1, cdf[pos]], c="lightgray", linewidth=2, linestyle="--", zorder=0)
            add_stroke_dot(ax, pos, cdf[pos], color=main_color, s=10, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
            ax.text(pos, cdf[pos], str(pos)+f'{self.cost_name}\n'+str(round(p*100))+'%',
                    weight='bold',
                    size=10,
                    color='gray',
                    transform=trans,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    path_effects=self.plot_path_effect)
        
        # Set description text
        description_text = self.description_func(
            item_name=self.item_name,
            cost_name=self.cost_name,
            text_head=self.text_head,
            mark_exp=self.dist_data.exp,
            show_max_pull=len(dist)-1,  
            is_finite=self.is_finite,
            text_tail=self.text_tail,
            )  
        if show_description:
            ax.text(
                self.description_pos, 1.033,
                description_text,
                weight='bold',
                size=12,
                color='#B0B0B0',
                path_effects=self.plot_path_effect,
                horizontalalignment='left',
                verticalalignment='top',
                zorder=11)

if __name__ == '__main__':
    import GGanalysis.games.genshin_impact as GI
    import time
    
    pass
    a = GI.common_5star(1)
    fig = DrawDistribution(
        a,
        item_name='five star props',
        quantile_pos=[0.1, 0.2, 0.3, 0.5, 0.9],
        text_head='Adopt official public model',
        text_tail='@balanced pity '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    )
    fig.show_dist()
    # fig.show_cdf()
    # fig.show_two_graph()
    