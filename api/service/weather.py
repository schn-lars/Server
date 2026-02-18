from service.requestforms import HourlyRequest, MinutelyRequest
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import io

def get_hourly_weather(request: HourlyRequest):
    lang = request.language
    temperatures = [hour.degrees for hour in request.slots]
    rain = [hour.millimeters for hour in request.slots]
    rain_chance = [hour.chance for hour in request.slots]
    hours = [hour.hours for hour in request.slots]
    positions = list(range(len(hours)))

    fig, ax = plt.subplots(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.05, 0.95], wspace=0.05)
    cbar_ax = fig.add_axes([0.05, 0.15, 0.03, 0.7]) # color is the 3rd
    #cbar_ax = fig.add_subplot(gs[0])
    #ax = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.2)
    ax.bar(positions, rain, color='lightskyblue', label='Precipitation' if lang == 'ENG' else 'Niederschlag')

    degree_ax = ax.twinx()
    degree_ax.plot(positions, temperatures, '-r', label='Temperature' if lang == 'ENG' else 'Temperatur')

    ax.set_xticks(positions)
    ax.set_xticklabels(hours)

    max_rain = max(rain)
    y_max = 10 if max_rain <= 9 else max_rain * 1.1
    ax.set_ylim(0, y_max)

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap("Blues")

    for pos, chance in zip(positions, rain_chance):
        color = cmap(chance)
        ax.add_patch(plt.Rectangle(
            (pos - 0.4, y_max + 0.2),
            0.8,
            0.3,
            color=color,
            transform=ax.transData,
            clip_on=False
        ))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Rain Chance (%)" if lang == "ENG" else "Regen Wahrscheinlichkeit (%)", rotation=270)
    cbar.set_ticks(np.linspace(0, 1, 11))
    cbar.set_ticklabels([f"{int(p * 100)}%" for p in np.linspace(0, 1, 11)])

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = degree_ax.get_legend_handles_labels()

    smallest_temp_index = temperatures.index(min(temperatures))
    if 0 <= smallest_temp_index < 9:
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    elif 9 <= smallest_temp_index < 17:
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper center')
    else:
        ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    ax.grid()

    ax.set_xlabel('Hours' if lang == "ENG" else "Stunden") # or better the actual day "Di, DD.MM."
    ax.set_ylabel("mm")
    degree_ax.set_ylabel("°C")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def get_minutely_weather(request: MinutelyRequest):
    lang = request.language
    rain_chance = [slot.chance for slot in request.slots]
    minutes = [slot.minute for slot in request.slots]

    fig, ax = plt.subplots()
    ax.plot(minutes, rain_chance, marker='o')
    ax.set_xlabel("Minutes" if lang == "ENG" else "Minuten")
    ax.set_ylabel("Rain Chance (%)" if lang == "ENG" else "Regen Wahrscheinlichkeit (%)")
    ax.set_xticks(minutes[::5])
    ax.set_ylim(0, 1)
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf