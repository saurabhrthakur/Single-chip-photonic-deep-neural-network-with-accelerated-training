import gdsfactory as gf
import uuid

# ==============================================================================
# SECTION 1: GDSFACTORY LAYOUT COMPONENTS
# ==============================================================================

@gf.cell
def basic_waveguide(length: float = 10.0, width: float = 0.5, layer: tuple = (1, 0)) -> gf.Component:
    """Creates a basic straight waveguide."""
    c = gf.Component(f"waveguide_{uuid.uuid4().hex[:8]}")
    c.add_polygon([(0, -width/2), (length, -width/2), (length, width/2), (0, width/2)], layer=layer)
    c.add_port(name="o1", center=(0, 0), width=width, orientation=180, layer=layer)
    c.add_port(name="o2", center=(length, 0), width=width, orientation=0, layer=layer)
    return c

@gf.cell
def directional_coupler(
    length: float = 20.0,
    gap: float = 0.2,
    width: float = 0.5,
    layer: tuple = (1, 0)
) -> gf.Component:
    """Creates a directional coupler with two parallel waveguides."""
    c = gf.Component(f"coupler_{uuid.uuid4().hex[:8]}")
    wg_top = c << basic_waveguide(length=length, width=width, layer=layer)
    wg_bot = c << basic_waveguide(length=length, width=width, layer=layer)
    wg_top.movey(width/2 + gap/2)
    wg_bot.movey(-(width/2 + gap/2))
    c.add_port("o1", port=wg_bot.ports["o1"])
    c.add_port("o2", port=wg_top.ports["o1"])
    c.add_port("o3", port=wg_bot.ports["o2"])
    c.add_port("o4", port=wg_top.ports["o2"])
    return c

@gf.cell
def phase_shifter(
    length: float = 10.0, width: float = 0.5, layer: tuple = (1, 0),
    heater_layer: tuple = (2, 0), heater_width: float = 1.0, heater_offset: float = 1.0
) -> gf.Component:
    """Creates a phase shifter with a heater element."""
    c = gf.Component(f"phaseshifter_{uuid.uuid4().hex[:8]}")
    wg = c << basic_waveguide(length=length, width=width, layer=layer)
    heater = c << gf.components.rectangle(
        size=(length, heater_width),
        layer=heater_layer
    )
    heater.movey(width/2 + heater_offset + heater_width/2)
    c.add_ports(wg.ports)
    return c

@gf.cell
def mzi(
    coupler_length: float = 20.0, coupler_gap: float = 0.2, arm_length: float = 40.0,
    phase_shifter_length: float = 10.0, width: float = 0.5, layer: tuple = (1, 0),
    heater_layer: tuple = (2, 0), heater_width: float = 1.0, heater_offset: float = 1.0
) -> gf.Component:
    """Creates a Mach-Zehnder Interferometer (MZI)."""
    c = gf.Component(f"mzi_{uuid.uuid4().hex[:8]}")
    dc1 = c << directional_coupler(length=coupler_length, gap=coupler_gap, width=width, layer=layer)
    dc2 = c << directional_coupler(length=coupler_length, gap=coupler_gap, width=width, layer=layer)
    ps = c << phase_shifter(length=phase_shifter_length, width=width, layer=layer,
                             heater_layer=heater_layer, heater_width=heater_width, heater_offset=heater_offset)
    
    arm_top = c << basic_waveguide(length=arm_length, width=width, layer=layer)
    arm_bot = c << basic_waveguide(length=arm_length, width=width, layer=layer)

    arm_top.connect("o1", dc1.ports["o4"])
    ps.connect("o1", arm_top.ports["o2"])
    dc2.connect("o4", ps.ports["o2"])
    
    arm_bot.connect("o1", dc1.ports["o3"])
    dc2.connect("o3", arm_bot.ports["o2"])

    c.add_port("o1", port=dc1.ports["o1"])
    c.add_port("o2", port=dc1.ports["o2"])
    c.add_port("o3", port=dc2.ports["o1"])
    c.add_port("o4", port=dc2.ports["o2"])
    
    c.add_port("e1", port=ps.ports["e1"])
    c.add_port("e2", port=ps.ports["e2"])
    
    return c

@gf.cell
def photodetector(
    name: str = "photodetector",
    width: float = 10.0,
    height: float = 10.0,
    layer: tuple = (2, 0),
    port_width: float = 0.5,
) -> gf.Component:
    """Creates a simple photodetector component."""
    c = gf.Component(name + f"_{uuid.uuid4().hex[:8]}")
    pd = c << gf.components.rectangle(size=(width, height), layer=layer)
    c.add_port(
        name="o1",
        center=(pd.xmin, pd.center[1]),
        width=port_width,
        orientation=180,
        layer=layer,
    )
    c.info["responsivity"] = 0.8  # A/W
    return c

@gf.cell
def microring_with_heater_pads(
    radius: float = 10.0, gap: float = 0.2, **kwargs
) -> gf.Component:
    """Creates a microring with heater pads for tuning."""
    c = gf.components.ring_single_heater(radius=radius, gap=gap, **kwargs)
    return c

def mzi_tap(**kwargs) -> gf.Component:
    """A simple MZI used as a tap or modulator."""
    c = gf.components.mzi_phase_shifter(delta_length=10.0, length_y=2.0, **kwargs)
    return c

@gf.cell
def nofu_reconfigurable_layout(**kwargs) -> gf.Component:
    """Assembles the layout for the reconfigurable NOFU."""
    c = gf.Component(f"nofu_reconfigurable_{uuid.uuid4().hex[:8]}")
    mzi_comp = mzi_tap()
    ring = microring_with_heater_pads(radius=10)
    pd_comp = gf.components.straight(length=15, width=2)

    mzi_ref = c.add_ref(mzi_comp, "mzi_tap")
    ring_ref = c.add_ref(ring, "microring")
    pd_ref = c.add_ref(pd_comp, "photodiode")

    ring_ref.connect("o1", mzi_ref.ports["o2"])
    pd_ref.connect("o1", mzi_ref.ports["o1"])

    c.add_port("optical_in", port=mzi_ref.ports["o3"])
    c.add_port("optical_out", port=ring_ref.ports["o2"])
    c.add_port("e_beta", port=mzi_ref.ports["e2"])
    c.add_port("e_detuning", port=ring_ref.ports["e2"])
    return c

def splitter_tree_1x6(splitter_func=gf.components.mmi1x2, **kwargs) -> gf.Component:
    c = gf.Component(f"splitter_tree_1x6_{uuid.uuid4().hex[:8]}")
    s1 = c << splitter_func(**kwargs)
    s2 = c << splitter_func(**kwargs)
    s3 = c << splitter_func(**kwargs)
    s4 = c << splitter_func(**kwargs)
    s5 = c << splitter_func(**kwargs)

    s2.connect("o1", s1.ports["o2"])
    s3.connect("o1", s1.ports["o3"])
    s4.connect("o1", s2.ports["o3"])
    s5.connect("o1", s3.ports["o3"])
    
    c.add_port("o1", port=s1.ports["o1"])
    c.add_port("o2", port=s2.ports["o2"])
    c.add_port("o3", port=s4.ports["o2"])
    c.add_port("o4", port=s4.ports["o3"])
    c.add_port("o5", port=s5.ports["o2"])
    c.add_port("o6", port=s5.ports["o3"])
    return c

@gf.cell
def ict_realistic(
    n_channels=6,
    splitter_func=splitter_tree_1x6,
    modulator_func=mzi,
    **kwargs
) -> gf.Component:
    c = gf.Component(f"ict_realistic_{uuid.uuid4().hex[:8]}")
    splitter = c << splitter_func(**kwargs)
    
    for i in range(n_channels):
        mod = c << modulator_func(**kwargs)
        mod.connect("o1", splitter.ports[f"o{i+1}"])
        c.add_port(f"optical_out_{i+1}", port=mod.ports["o3"])
        c.add_port(f"elec_in_{i+1}", port=mod.ports["e1"])
        
    c.add_port("optical_in", port=splitter.ports["o1"])
    return c

@gf.cell
def cmxu_6x6(n=6, **kwargs):
    """Creates a 6x6 CMXU using a Clements mesh of MZIs."""
    c = gf.Component(f"cmxu_{n}x{n}_{uuid.uuid4().hex[:8]}")
    mzis = [[c << mzi(**kwargs) for _ in range(n - i)] for i in range(n)]

    for i in range(n):
        for j in range(n - i):
            mzi_ref = mzis[i][j]
            y_pos = i * 40 + j * 80
            x_pos = i * 100
            mzi_ref.move((x_pos, y_pos))

    # This connection logic is complex and would need to be carefully implemented
    # based on the Clements architecture diagram. For now, it's a placeholder.

    # Example of connecting a few MZIs
    if n > 1:
        mzis[0][1].connect("o1", mzis[0][0].ports["o3"])
        if n > 2:
             mzis[1][0].connect("o2", mzis[0][0].ports["o4"])

    # Expose ports
    for i in range(n):
        c.add_port(f"in_{i}", port=mzis[0][i].ports["o1"])
        c.add_port(f"out_{i}", port=mzis[i][0].ports["o4"]) # This is an approximation
        
    return c

def add_route_to_component(c, route):
    if isinstance(route, gf.Component) or isinstance(route, gf.ComponentReference):
        c.add(route)
    elif isinstance(route, (list, tuple)):
        for r in route:
            add_route_to_component(c, r)

@gf.cell
def build_full_onn(
    n_layers=3, n_channels=6, cmxu_func=cmxu_6x6, cmxu_kwargs=None,
    nofu_func=nofu_reconfigurable_layout, nofu_kwargs=None,
    ict_func=ict_realistic, ict_kwargs=None, spacing=20.0
) -> gf.Component:
    """
    Placeholder for the full ONN layout assembly.
    This function is complex and requires careful routing.
    """
    # This function is not fully implemented due to routing complexities.
    # It remains here as a template for future layout work.
    c = gf.Component("full_onn_placeholder")
    text = c << gf.components.text(
        "Full ONN Layout requires\ncomplex manual routing.\nThis is a placeholder."
    )
    return c 