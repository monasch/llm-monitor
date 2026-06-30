import matplotlib

column_width = 3.25      # in — single column figure
text_width = 6.75        # in — figure* spanning both columns
dpi = 300

fs_m1 = 6  # for figure ticks
fs = 8  # for regular figure text
fs_p1 = 9  #  figure titles

matplotlib.rc("font", size=fs)  # controls default text sizes
matplotlib.rc("axes", titlesize=fs)  # fontsize of the axes title
matplotlib.rc("axes", labelsize=fs)  # fontsize of the x and y labels
matplotlib.rc("xtick", labelsize=fs_m1)  # fontsize of the tick labels
matplotlib.rc("ytick", labelsize=fs_m1)  # fontsize of the tick labels
matplotlib.rc("legend", fontsize=fs_m1)  # legend fontsize
matplotlib.rc("figure", titlesize=fs_p1, dpi=dpi, autolayout=True)  # fontsize of the figure
matplotlib.rc("lines", linewidth=1, markersize=3)
matplotlib.rc("savefig", dpi=1200, bbox="tight")
matplotlib.rc("grid", alpha=0.3)
matplotlib.rc("axes", grid=True)
matplotlib.rc("font", family="serif", serif=["Times New Roman", "Times", "DejaVu Serif"])
matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",   # Times-compatible math, not "cm"
})
matplotlib.rc("text", usetex=False)