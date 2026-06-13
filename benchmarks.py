"""Benchmark fixtures for the direct-vs-ToT comparison harnesses.

This module intentionally lives OUTSIDE the runtime skill layer: it contains
the benchmark problems, expected values, reference answers, and per-case tool
solutions. Keeping it out of ``skills.SKILL_REGISTRY`` ensures the system
under test cannot discover benchmark answers through skill search at runtime.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import sympy as sp


def benchmark_symbolic_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    """Solve structured symbolic/integer benchmark operations for tool-assisted ToT comparisons."""

    operation = str(params.get("operation", "")).strip().lower()
    if not operation:
        raise ValueError("benchmark_symbolic_solver requires an operation.")

    def _final(value: Any) -> Dict[str, Any]:
        simplified = sp.simplify(value)
        if getattr(simplified, "is_Integer", False):
            answer = str(int(simplified))
        elif getattr(simplified, "is_Rational", False):
            answer = f"{float(simplified):.12g}"
        else:
            answer = str(simplified)
        return {
            "final_answer": answer,
            "concise_solution": f"benchmark_symbolic_solver:{operation}",
            "raw_value": str(simplified),
        }

    if operation == "determinant":
        return _final(sp.Matrix(params["matrix"]).det())

    if operation == "matrix_power_entry":
        matrix = sp.Matrix(params["matrix"])
        power = int(params["power"])
        row = int(params.get("row", 1)) - 1
        col = int(params.get("col", 1)) - 1
        return _final((matrix**power)[row, col])

    if operation == "linear_recurrence_mod":
        value = int(params["initial"])
        multiplier = int(params["multiplier"])
        increment = int(params["increment"])
        modulus = int(params["modulus"])
        steps = int(params["steps"])
        for _ in range(steps):
            value = (multiplier * value + increment) % modulus
        return _final(value)

    if operation == "derivative_value":
        variable = sp.Symbol(str(params.get("variable", "x")))
        expression = sp.sympify(str(params["expression"]))
        order = int(params.get("order", 1))
        at_value = sp.sympify(str(params["at"]))
        return _final(sp.diff(expression, variable, order).subs(variable, at_value))

    if operation == "polynomial_coefficient":
        variable = sp.Symbol(str(params.get("variable", "x")))
        expression = sp.expand(sp.sympify(str(params["expression"])))
        power = int(params["power"])
        return _final(expression.coeff(variable, power))

    raise ValueError(f"Unsupported benchmark_symbolic_solver operation: {operation}")


def benchmark_physics_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    """Solve structured physics benchmark operations for tool-assisted ToT comparisons."""

    operation = str(params.get("operation", "")).strip().lower()
    if not operation:
        raise ValueError("benchmark_physics_solver requires an operation.")

    def _final(value: Any, unit: str = "") -> Dict[str, Any]:
        simplified = sp.simplify(value)
        if getattr(simplified, "is_Integer", False):
            answer = str(int(simplified))
        elif getattr(simplified, "is_Rational", False):
            answer = f"{float(simplified):.12g}"
        else:
            answer = str(simplified)
        unit_text = str(unit).strip()
        if unit_text:
            answer = f"{answer} {unit_text}"
        return {
            "final_answer": answer,
            "concise_solution": f"benchmark_physics_solver:{operation}",
            "raw_value": str(simplified),
        }

    if operation == "linearized_eigenvalue_product":
        matrix = sp.Matrix(params["operator_matrix"])
        return _final(matrix.det(), str(params.get("unit", "")))

    if operation == "first_static_instability":
        load = sp.Symbol(str(params.get("load_symbol", "p")))
        matrix = sp.Matrix(params["stiffness_matrix"])
        slope = sp.Matrix(params["load_slope_matrix"])
        determinant = sp.factor((matrix + load * slope).det())
        roots = sp.nroots(determinant)
        tolerance = float(params.get("imag_tol", 1e-8))
        positive_roots = sorted(
            float(sp.re(root))
            for root in roots
            if abs(float(sp.im(root))) <= tolerance and float(sp.re(root)) > 0
        )
        if not positive_roots:
            raise ValueError("No positive real static-instability root found.")
        return _final(sp.nsimplify(positive_roots[0]), str(params.get("unit", "")))

    if operation == "linear_solve_component":
        matrix = sp.Matrix(params["coefficient_matrix"])
        rhs_vector = sp.Matrix(params["rhs_vector"])
        component_index = int(params.get("component_index", 0))
        solution = matrix.LUsolve(rhs_vector)
        if component_index < 0 or component_index >= len(solution):
            raise ValueError("component_index is outside the linear-system solution vector.")
        return _final(solution[component_index], str(params.get("unit", "")))

    if operation == "lowest_generalized_frequency":
        stiffness_matrix = sp.Matrix(params["stiffness_matrix"])
        mass_matrix = sp.Matrix(params["mass_matrix"])
        eigenvalue = sp.Symbol(str(params.get("eigenvalue_symbol", "lambda")))
        characteristic = sp.factor((stiffness_matrix - eigenvalue * mass_matrix).det())
        roots = sp.nroots(characteristic)
        tolerance = float(params.get("imag_tol", 1e-8))
        positive_roots = sorted(
            float(sp.re(root))
            for root in roots
            if abs(float(sp.im(root))) <= tolerance and float(sp.re(root)) > 0
        )
        if not positive_roots:
            raise ValueError("No positive generalized-frequency eigenvalue found.")
        return _final(sp.N(sp.sqrt(positive_roots[0]), 15), str(params.get("unit", "")))

    if operation == "magnetized_cylinder_axis_field":
        mu0_magnetization = sp.sympify(params["mu0_magnetization"])
        radius = sp.sympify(params["radius"])
        length = sp.sympify(params["length"])
        axial_position = sp.sympify(params["axial_position"])
        upper_offset = axial_position + length / 2
        lower_offset = axial_position - length / 2
        field = mu0_magnetization / 2 * (
            upper_offset / sp.sqrt(radius**2 + upper_offset**2)
            - lower_offset / sp.sqrt(radius**2 + lower_offset**2)
        )
        return _final(sp.N(field, 15), str(params.get("unit", "")))

    if operation == "matrix_product_half_trace":
        matrices = [sp.Matrix(item) for item in params["matrices"]]
        if not matrices:
            raise ValueError("matrix_product_half_trace requires at least one matrix.")
        product = sp.eye(matrices[0].rows)
        for matrix in matrices:
            product = matrix * product
        return _final(sp.trace(product) / 2, str(params.get("unit", "")))

    if operation == "relativistic_threshold_kinetic_energy":
        projectile_mass = sp.sympify(params["projectile_mass"])
        target_mass = sp.sympify(params["target_mass"])
        final_masses = [sp.sympify(item) for item in params["final_masses"]]
        final_rest_mass = sum(final_masses, sp.Integer(0))
        projectile_total_energy = (
            final_rest_mass**2 - projectile_mass**2 - target_mass**2
        ) / (2 * target_mass)
        threshold_kinetic_energy = projectile_total_energy - projectile_mass
        return _final(sp.N(threshold_kinetic_energy, 15), str(params.get("unit", "")))

    if operation == "rough_pipe_pressure_drop":
        density = float(params["density"])
        viscosity = float(params["dynamic_viscosity"])
        diameter = float(params["diameter"])
        length = float(params["length"])
        roughness = float(params["roughness"])
        flow_rate = float(params["flow_rate"])
        velocity = flow_rate / (math.pi * diameter**2 / 4)
        reynolds = density * velocity * diameter / viscosity
        relative_roughness = roughness / diameter

        def colebrook_residual(friction_factor: float) -> float:
            return 1 / math.sqrt(friction_factor) + 2 * math.log10(
                relative_roughness / 3.7
                + 2.51 / (reynolds * math.sqrt(friction_factor))
            )

        low = 0.008
        high = 0.12
        for _ in range(100):
            midpoint = (low + high) / 2
            if colebrook_residual(low) * colebrook_residual(midpoint) <= 0:
                high = midpoint
            else:
                low = midpoint
        friction_factor = (low + high) / 2
        pressure_drop = friction_factor * (length / diameter) * density * velocity**2 / 2
        return {
            "final_answer": f"{friction_factor:.12g} friction factor, {pressure_drop:.12g} Pa",
            "concise_solution": f"benchmark_physics_solver:{operation}",
            "raw_value": f"friction_factor={friction_factor}; pressure_drop={pressure_drop}; Re={reynolds}",
        }

    if operation == "isentropic_nozzle_supersonic":
        gamma = float(params.get("gamma", 1.4))
        area_ratio = float(params["area_ratio"])

        def area_mach(mach: float) -> float:
            term = 2 / (gamma + 1) * (1 + (gamma - 1) * mach**2 / 2)
            exponent = (gamma + 1) / (2 * (gamma - 1))
            return term**exponent / mach

        low = 1.000001
        high = 20.0
        for _ in range(120):
            midpoint = (low + high) / 2
            if area_mach(midpoint) < area_ratio:
                low = midpoint
            else:
                high = midpoint
        mach = (low + high) / 2
        pressure_ratio = (1 + (gamma - 1) * mach**2 / 2) ** (-gamma / (gamma - 1))
        return {
            "final_answer": f"{mach:.12g} Mach, {pressure_ratio:.12g} pressure ratio",
            "concise_solution": f"benchmark_physics_solver:{operation}",
            "raw_value": f"Mach={mach}; pressure_ratio={pressure_ratio}",
        }

    if operation == "hydraulic_jump_rectangular":
        upstream_depth = float(params["upstream_depth"])
        upstream_velocity = float(params["upstream_velocity"])
        gravity = float(params.get("gravity", 9.81))
        froude = upstream_velocity / math.sqrt(gravity * upstream_depth)
        downstream_depth = upstream_depth / 2 * (math.sqrt(1 + 8 * froude**2) - 1)
        energy_loss = (downstream_depth - upstream_depth) ** 3 / (
            4 * upstream_depth * downstream_depth
        )
        return {
            "final_answer": f"{downstream_depth:.12g} m, {energy_loss:.12g} m energy loss",
            "concise_solution": f"benchmark_physics_solver:{operation}",
            "raw_value": f"Fr1={froude}; y2={downstream_depth}; energy_loss={energy_loss}",
        }

    if operation == "laminar_flat_plate_drag":
        density = float(params["density"])
        viscosity = float(params["dynamic_viscosity"])
        velocity = float(params["velocity"])
        length = float(params["length"])
        width = float(params["width"])
        reynolds_length = density * velocity * length / viscosity
        average_friction_coefficient = 1.328 / math.sqrt(reynolds_length)
        drag = 0.5 * density * velocity**2 * length * width * average_friction_coefficient
        return {
            "final_answer": f"{average_friction_coefficient:.12g} average Cf, {drag:.12g} N",
            "concise_solution": f"benchmark_physics_solver:{operation}",
            "raw_value": f"Re_L={reynolds_length}; Cf={average_friction_coefficient}; drag={drag}",
        }

    if operation == "capillary_gravity_phase_speed":
        density = float(params["density"])
        surface_tension = float(params["surface_tension"])
        depth = float(params["depth"])
        wavelength = float(params["wavelength"])
        gravity = float(params.get("gravity", 9.81))
        wave_number = 2 * math.pi / wavelength
        angular_frequency_squared = (
            gravity * wave_number + surface_tension / density * wave_number**3
        ) * math.tanh(wave_number * depth)
        phase_speed = math.sqrt(angular_frequency_squared) / wave_number
        return {
            "final_answer": f"{phase_speed:.12g} m/s",
            "concise_solution": f"benchmark_physics_solver:{operation}",
            "raw_value": f"k={wave_number}; phase_speed={phase_speed}",
        }

    raise ValueError(f"Unsupported benchmark_physics_solver operation: {operation}")


DIRECT_9B_BENCHMARK_DATA: Dict[str, Any] = {
    "default_suite": "ap1",
    "system_prompt": (
        "You are a careful quantitative problem solver. Return only a JSON object with keys "
        "final_answer and concise_solution. The final_answer must contain the final numeric value(s) with units. "
        "Do not use markdown."
    ),
    "freeform_system_prompt": (
        "You are a careful quantitative problem solver. Solve naturally without using a fixed JSON schema. "
        "Show enough work to justify the answer, and put the final numeric answer with units at the end."
    ),
    "cases": {
        "ap1": [
            {
                "case_id": "angled_pull_with_friction",
                "topic": "forces_and_kinematics",
                "prompt": "A 5.0 kg crate on a horizontal floor is pulled by a 30 N force at 30 degrees above the horizontal. The coefficient of kinetic friction is 0.20. Starting from rest, how far does it move in 4.0 s? Use g = 9.8 m/s^2.",
                "expected_values": [{"label": "distance_m", "value": 30.7, "abs_tol": 0.4, "rel_tol": 0.03}],
                "reference_answer": "x = 0.5*((30*cos30 - 0.20*(5*9.8 - 30*sin30))/5)*4^2 = 30.7 m",
            },
            {
                "case_id": "atwood_acceleration_tension",
                "topic": "newton_second_law_systems",
                "prompt": "An ideal Atwood machine has masses 3.0 kg and 5.0 kg connected by a light string over a frictionless pulley. Find the magnitude of the acceleration and the string tension. Use g = 9.8 m/s^2.",
                "expected_values": [
                    {"label": "acceleration_m_s2", "value": 2.45, "abs_tol": 0.08, "rel_tol": 0.03},
                    {"label": "tension_N", "value": 36.8, "abs_tol": 0.6, "rel_tol": 0.03},
                ],
                "reference_answer": "a = (5 - 3)*9.8/(5 + 3) = 2.45 m/s^2; T = 3*(9.8 + 2.45) = 36.8 N",
            },
            {
                "case_id": "loop_top_normal_force",
                "topic": "energy_and_circular_motion",
                "prompt": "A 1.0 kg cart starts from rest at height 3R above the bottom of a frictionless vertical loop of radius R. Find the normal force at the top of the loop. Use g = 9.8 m/s^2.",
                "expected_values": [{"label": "normal_force_N", "value": 9.8, "abs_tol": 0.3, "rel_tol": 0.03}],
                "reference_answer": "N_top = m*(2*g*(3R - 2R))/R - m*g = 9.8 N",
            },
            {
                "case_id": "inelastic_collision_spring",
                "topic": "momentum_and_energy",
                "prompt": "A 0.50 kg cart moving at 4.0 m/s sticks to a 1.50 kg cart initially at rest. The joined carts then compress a spring with k = 200 N/m. What is the maximum compression?",
                "expected_values": [{"label": "compression_m", "value": 0.100, "abs_tol": 0.02, "rel_tol": 0.03}],
                "reference_answer": "x_spring = sqrt(((0.50*4.0)/(0.50+1.50))^2*(0.50+1.50)/200) = 0.100 m",
            },
            {
                "case_id": "disk_angular_acceleration",
                "topic": "rotational_dynamics",
                "prompt": "A solid disk of mass 4.0 kg and radius 0.50 m is initially at rest. A tangential force of 10 N is applied at the rim for 3.0 s. Find the final angular speed.",
                "expected_values": [{"label": "angular_speed_rad_s", "value": 30.0, "abs_tol": 0.8, "rel_tol": 0.03}],
                "reference_answer": "omega = (10*0.50/(0.5*4.0*0.50^2))*3.0 = 30 rad/s",
            },
            {
                "case_id": "rolling_cylinder_down_ramp",
                "topic": "rotational_energy",
                "prompt": "A solid cylinder rolls without slipping from rest down a ramp through a vertical height of 2.0 m. Find its speed at the bottom. Use g = 9.8 m/s^2 and I = (1/2)MR^2.",
                "expected_values": [{"label": "speed_m_s", "value": 5.11, "abs_tol": 0.16, "rel_tol": 0.03}],
                "reference_answer": "v = sqrt(2*g*h/(1 + 1/2)) = 5.11 m/s",
            },
            {
                "case_id": "spring_shm_max_values",
                "topic": "simple_harmonic_motion",
                "prompt": "A 0.50 kg mass on a spring with k = 200 N/m oscillates with amplitude 0.10 m. Find the maximum speed and maximum acceleration.",
                "expected_values": [
                    {"label": "max_speed_m_s", "value": 2.0, "abs_tol": 0.12, "rel_tol": 0.03},
                    {"label": "max_acceleration_m_s2", "value": 40.0, "abs_tol": 1.2, "rel_tol": 0.03},
                ],
                "reference_answer": "v_max = A*sqrt(k/m) = 2.0 m/s; a_max = (k/m)*A = 40 m/s^2",
            },
            {
                "case_id": "pendulum_bottom_tension",
                "topic": "energy_and_circular_force",
                "prompt": "A 0.50 kg pendulum bob of length 1.20 m is released from rest at 30 degrees from vertical. Find the string tension at the bottom. Use g = 9.8 m/s^2.",
                "expected_values": [{"label": "tension_N", "value": 6.22, "abs_tol": 0.2, "rel_tol": 0.03}],
                "reference_answer": "T_bottom = m*g + m*(2*g*1.20*(1 - cos30))/1.20 = 6.22 N",
            },
            {
                "case_id": "impulse_force_time_graph",
                "topic": "impulse_and_momentum",
                "prompt": "A 2.0 kg object initially moves at 3.0 m/s. It experiences a triangular force-time pulse with peak 40 N and duration 0.20 s in the direction of motion. Find the final speed.",
                "expected_values": [{"label": "final_speed_m_s", "value": 5.0, "abs_tol": 0.12, "rel_tol": 0.03}],
                "reference_answer": "v_final = (2.0*3.0 + 0.5*0.20*40)/2.0 = 5.0 m/s",
            },
            {
                "case_id": "work_energy_force_position_graph",
                "topic": "work_energy_from_graph",
                "prompt": "A 2.0 kg block starts from rest. The force along its displacement rises linearly from 0 to 12 N over the first 3.0 m, then a constant 3.0 N friction force acts opposite the motion throughout those 3.0 m. Find the speed after 3.0 m.",
                "expected_values": [{"label": "speed_m_s", "value": 3.0, "abs_tol": 0.12, "rel_tol": 0.03}],
                "reference_answer": "v = sqrt(2*(0.5*3.0*12 - 3.0*3.0)/2.0) = 3.0 m/s",
            },
        ],
        "em": [
            {
                "case_id": "em_nonuniform_line_charge_axis_field",
                "topic": "physics_c_em_calculus",
                "prompt": "A thin rod lies on the x-axis from x=0 to x=2.0 m. Its linear charge density is lambda(x)=lambda0*x/L, where lambda0=4.0 nC/m and L=2.0 m. Find the magnitude of the electric field at x=-1.0 m in N/C. Use k=8.99e9 N m^2/C^2.",
                "expected_values": [{"label": "field_N_C", "value": 7.77, "abs_tol": 0.45, "rel_tol": 0.03}],
                "reference_answer": "|E| = (k*lambda0/L)*(ln((a+L)/a) + a/(a+L) - 1) = 7.77 N/C",
            },
            {
                "case_id": "em_uniform_disk_axis_field",
                "topic": "physics_c_em_calculus",
                "prompt": "A uniformly charged disk has surface charge density sigma=2.0 microC/m^2 and radius R=0.30 m. Find the electric field magnitude on the axis at z=0.40 m in N/C. Use epsilon0=8.854e-12 F/m.",
                "expected_values": [{"label": "field_N_C", "value": 2.26e4, "abs_tol": 1.1e3, "rel_tol": 0.03}],
                "reference_answer": "E = sigma/(2*epsilon0)*(1 - z/sqrt(z^2 + R^2)) = 2.26e4 N/C",
            },
            {
                "case_id": "em_radial_density_sphere_inside_field",
                "topic": "physics_c_em_calculus",
                "prompt": "A solid sphere of radius R=0.20 m has volume charge density rho(r)=rho0*r/R, where rho0=5.0 microC/m^3. Use Gauss's law to find the electric field magnitude at r=0.10 m in N/C. Use epsilon0=8.854e-12 F/m.",
                "expected_values": [{"label": "field_N_C", "value": 7.06e3, "abs_tol": 3.5e2, "rel_tol": 0.03}],
                "reference_answer": "E(r<R) = rho0*r^2/(4*epsilon0*R) = 7.06e3 N/C",
            },
            {
                "case_id": "em_exponential_field_potential",
                "topic": "physics_c_em_calculus",
                "prompt": "In one dimension, E(x)=E0*exp(-x/a) in the +x direction for x>=0, with E0=100 V/m and a=0.50 m. If V(infinity)=0, find V at x=1.0 m in volts.",
                "expected_values": [{"label": "potential_V", "value": 6.77, "abs_tol": 0.35, "rel_tol": 0.03}],
                "reference_answer": "V(x)=E0*a*exp(-x/a)=6.77 V",
            },
            {
                "case_id": "em_coaxial_capacitance_pf",
                "topic": "physics_c_em_calculus",
                "prompt": "A coaxial capacitor has inner radius a=1.0 mm, outer radius b=4.0 mm, and length L=0.50 m. Find its capacitance in pF. Use epsilon0=8.854e-12 F/m.",
                "expected_values": [{"label": "capacitance_pF", "value": 20.1, "abs_tol": 1.2, "rel_tol": 0.03}],
                "reference_answer": "C = 2*pi*epsilon0*L/ln(b/a) = 20.1 pF",
            },
            {
                "case_id": "em_rc_discharge_milli_units",
                "topic": "physics_c_em_calculus",
                "prompt": "A 100 microF capacitor initially charged to 12 V discharges through a 2.0 kOhm resistor. At t=0.40 s, find the remaining charge in mC and the current magnitude in mA.",
                "expected_values": [
                    {"label": "charge_mC", "value": 0.162, "abs_tol": 0.012, "rel_tol": 0.03},
                    {"label": "current_mA", "value": 0.812, "abs_tol": 0.06, "rel_tol": 0.03},
                ],
                "reference_answer": "Q=CV0*exp(-t/RC)=0.162 mC; |I|=(V0/R)*exp(-t/RC)=0.812 mA",
            },
            {
                "case_id": "em_nonuniform_current_wire_bfield_mt",
                "topic": "physics_c_em_calculus",
                "prompt": "A long cylindrical wire of radius R=1.0 cm has current density J(r)=J0*r/R, with J0=2.0e6 A/m^2. Use Ampere's law to find the magnetic field magnitude at r=0.50 cm in mT.",
                "expected_values": [{"label": "field_mT", "value": 2.09, "abs_tol": 0.14, "rel_tol": 0.03}],
                "reference_answer": "B(r<R)=mu0*J0*r^2/(3R)=2.09 mT",
            },
            {
                "case_id": "em_decaying_flux_emf_mv",
                "topic": "physics_c_em_calculus",
                "prompt": "A circular loop of radius 0.10 m is in a uniform perpendicular magnetic field B(t)=0.80*exp(-t/0.50) T. Find the induced emf magnitude at t=0.50 s in mV.",
                "expected_values": [{"label": "emf_mV", "value": 18.5, "abs_tol": 1.0, "rel_tol": 0.03}],
                "reference_answer": "|emf|=A*(B0/tau)*exp(-t/tau)=18.5 mV",
            },
        ],
        "traps": [
            {
                "case_id": "trap_constant_speed_incline_friction",
                "topic": "mixed_traps",
                "prompt": "A block slides down a 30 degree incline at constant speed. What is the coefficient of kinetic friction? Do not assume the normal force equals mg.",
                "expected_values": [{"label": "mu_k", "value": 0.577, "abs_tol": 0.035, "rel_tol": 0.03}],
                "reference_answer": "mu_k = tan(30 deg) = 0.577",
            },
            {
                "case_id": "trap_normal_force_work",
                "topic": "mixed_traps",
                "prompt": "A 2.0 kg bead moves 3.0 m along a frictionless circular wire while the wire's normal force is always perpendicular to the bead's instantaneous displacement. How much work does the normal force do? Answer in joules.",
                "expected_values": [{"label": "work_J", "value": 0.0, "abs_tol": 0.05, "rel_tol": 0.03}],
                "reference_answer": "W_N = 0 J because normal force is perpendicular to displacement at every instant",
            },
            {
                "case_id": "trap_loop_minimum_top_speed",
                "topic": "mixed_traps",
                "prompt": "At the top of a vertical loop of radius 2.0 m, what minimum speed must a cart have to just maintain contact? Use g=9.8 m/s^2. Answer in m/s.",
                "expected_values": [{"label": "speed_m_s", "value": 4.43, "abs_tol": 0.22, "rel_tol": 0.03}],
                "reference_answer": "v_min = sqrt(gR) = 4.43 m/s",
            },
            {
                "case_id": "trap_table_block_hanging_mass_friction",
                "topic": "mixed_traps",
                "prompt": "A 4.0 kg block on a horizontal table is connected over a pulley to a hanging 2.0 kg mass. The table has kinetic friction coefficient 0.20. Find the acceleration magnitude in m/s^2 and the string tension in N. Use g=9.8 m/s^2.",
                "expected_values": [
                    {"label": "acceleration_m_s2", "value": 1.96, "abs_tol": 0.12, "rel_tol": 0.03},
                    {"label": "tension_N", "value": 15.7, "abs_tol": 0.7, "rel_tol": 0.03},
                ],
                "reference_answer": "a=(m2*g-mu*m1*g)/(m1+m2)=1.96 m/s^2; T=m2*(g-a)=15.7 N",
            },
            {
                "case_id": "trap_potential_zero_not_field_zero",
                "topic": "mixed_traps",
                "prompt": "A +3.0 microC charge is fixed at x=0 and a -1.0 microC charge is fixed at x=1.0 m. Where to the right of x=1.0 m is the electric potential zero? Answer the x-coordinate in meters.",
                "expected_values": [{"label": "x_m", "value": 1.50, "abs_tol": 0.08, "rel_tol": 0.03}],
                "reference_answer": "3/x - 1/(x-1)=0 gives x=1.50 m",
            },
            {
                "case_id": "trap_induction_current_magnitude",
                "topic": "mixed_traps",
                "prompt": "A loop of area 0.20 m^2 and resistance 5.0 ohm sits in a perpendicular magnetic field that increases uniformly from 0.20 T to 0.80 T in 3.0 s. Find the induced current magnitude in mA.",
                "expected_values": [{"label": "current_mA", "value": 8.0, "abs_tol": 0.5, "rel_tol": 0.03}],
                "reference_answer": "I = A*DeltaB/(R*Deltat) = 8.0 mA",
            },
        ],
        "freeform": [
            {
                "case_id": "freeform_flux_derivative_emf",
                "topic": "freeform_no_schema",
                "prompt": "A magnetic field through a 0.050 m^2 loop is B(t)=0.40 t^2 tesla, perpendicular to the loop. Find the magnitude of the induced emf at t=3.0 s in volts.",
                "expected_values": [{"label": "emf_V", "value": 0.120, "abs_tol": 0.012, "rel_tol": 0.03}],
                "reference_answer": "|emf|=A*dB/dt=0.050*(0.80*3.0)=0.120 V",
                "freeform_output": True,
            },
            {
                "case_id": "freeform_uniform_rod_axis_field",
                "topic": "freeform_no_schema",
                "prompt": "A uniformly charged rod of length 0.50 m has total charge 5.0 nC. A point is on the rod's axis, 0.20 m from the nearer end. Find the electric field magnitude at that point in N/C. Use k=8.99e9.",
                "expected_values": [{"label": "field_N_C", "value": 321.0, "abs_tol": 18.0, "rel_tol": 0.03}],
                "reference_answer": "E=kQ/(a(a+L))=321 N/C",
                "freeform_output": True,
            },
            {
                "case_id": "freeform_loop_center_field_microt",
                "topic": "freeform_no_schema",
                "prompt": "A circular loop of radius 5.0 cm carries a current of 2.0 A. Find the magnetic field at its center in microtesla.",
                "expected_values": [{"label": "field_microT", "value": 25.1, "abs_tol": 1.5, "rel_tol": 0.03}],
                "reference_answer": "B=mu0*I/(2R)=25.1 microT",
                "freeform_output": True,
            },
            {
                "case_id": "freeform_acceleration_integral_displacement",
                "topic": "freeform_no_schema",
                "prompt": "A particle has acceleration a(t)=6t m/s^2, initial velocity v(0)=2 m/s, and initial position x(0)=1 m. Find x at t=2.0 s.",
                "expected_values": [{"label": "position_m", "value": 13.0, "abs_tol": 0.4, "rel_tol": 0.03}],
                "reference_answer": "v=3t^2+2; x=t^3+2t+1; x(2)=13 m",
                "freeform_output": True,
            },
            {
                "case_id": "freeform_rc_charging_voltage",
                "topic": "freeform_no_schema",
                "prompt": "A capacitor charges toward 12 V through R=2.0 kOhm and C=50 microF. Starting uncharged, what is the capacitor voltage at t=0.10 s?",
                "expected_values": [{"label": "voltage_V", "value": 7.59, "abs_tol": 0.35, "rel_tol": 0.03}],
                "reference_answer": "V=12*(1-exp(-t/RC))=7.59 V",
                "freeform_output": True,
            },
            {
                "case_id": "freeform_spring_launch_speed",
                "topic": "freeform_no_schema",
                "prompt": "A 2.0 kg block is launched on a frictionless table by a spring with k=500 N/m compressed by 0.20 m. Find the block's speed after leaving the spring.",
                "expected_values": [{"label": "speed_m_s", "value": 3.16, "abs_tol": 0.16, "rel_tol": 0.03}],
                "reference_answer": "0.5*k*x^2=0.5*m*v^2, so v=sqrt(k*x^2/m)=3.16 m/s",
                "freeform_output": True,
            },
        ],
        "limit": [
            {
                "case_id": "limit_signed_potential_difference",
                "topic": "limit_sign_and_calculus",
                "prompt": "Along the x-axis, the electric field is E_x(x)=4x+3 V/m from x=1.0 m to x=3.0 m. Find the signed potential difference V(3.0 m)-V(1.0 m) in volts.",
                "expected_values": [{"label": "delta_V", "value": -22.0, "abs_tol": 0.8, "rel_tol": 0.03}],
                "reference_answer": "V(3)-V(1)=-integral_1^3(4x+3)dx=-22 V",
            },
            {
                "case_id": "limit_reversed_battery_rc",
                "topic": "limit_transients",
                "prompt": "A 100 microF capacitor initially has voltage -5.0 V. At t=0 it is connected through a 2.0 kOhm resistor to an ideal +10.0 V battery. Find the capacitor voltage in V and the charging current magnitude in mA at t=0.10 s.",
                "expected_values": [
                    {"label": "capacitor_voltage_V", "value": 0.902, "abs_tol": 0.08, "rel_tol": 0.03},
                    {"label": "current_mA", "value": 4.55, "abs_tol": 0.25, "rel_tol": 0.03},
                ],
                "reference_answer": "Vc=10+(-15)exp(-0.10/0.20)=0.902 V; I=(10-Vc)/2000=4.55 mA",
            },
            {
                "case_id": "limit_coax_two_regions",
                "topic": "limit_ampere_piecewise",
                "prompt": "A coaxial cable has an inner solid conductor of radius 1.0 mm carrying 8.0 A uniformly out of the page. A thin outer cylindrical shell at radius 5.0 mm carries 8.0 A into the page. Find the magnetic field magnitudes at r=0.50 mm and r=4.0 mm in mT.",
                "expected_values": [
                    {"label": "B_inside_inner_mT", "value": 0.800, "abs_tol": 0.06, "rel_tol": 0.03},
                    {"label": "B_between_conductors_mT", "value": 0.400, "abs_tol": 0.04, "rel_tol": 0.03},
                ],
                "reference_answer": "B(0.50mm)=mu0*I*r/(2*pi*a^2)=0.800 mT; B(4.0mm)=mu0*I/(2*pi*r)=0.400 mT",
            },
            {
                "case_id": "limit_moving_loop_changing_field",
                "topic": "limit_flux_product_rule",
                "prompt": "A rectangular loop has fixed width 0.50 m. Its sliding side is at x(t)=0.20+0.30t meters, and a perpendicular field through the loop is B(t)=0.40+0.20t tesla. Find the induced emf magnitude at t=2.0 s in volts.",
                "expected_values": [{"label": "emf_V", "value": 0.200, "abs_tol": 0.02, "rel_tol": 0.03}],
                "reference_answer": "|emf|=d(B*w*x)/dt=w*(B' x+B x')=0.20 V at t=2.0 s",
            },
            {
                "case_id": "limit_disconnected_capacitor_dielectric",
                "topic": "limit_capacitor_energy",
                "prompt": "A 4.0 microF parallel-plate capacitor is charged to 100 V and then disconnected from the battery. A dielectric with kappa=3.0 is fully inserted. Find the final voltage in V and the energy lost in mJ.",
                "expected_values": [
                    {"label": "final_voltage_V", "value": 33.3, "abs_tol": 1.2, "rel_tol": 0.03},
                    {"label": "energy_lost_mJ", "value": 13.3, "abs_tol": 0.8, "rel_tol": 0.03},
                ],
                "reference_answer": "Vf=Vi/kappa=33.3 V; Ui=20.0 mJ, Uf=6.67 mJ, lost=13.3 mJ",
            },
            {
                "case_id": "limit_nonuniform_rod_rotation",
                "topic": "limit_nonuniform_inertia",
                "prompt": "A rod of length 2.0 m has total mass 3.0 kg and nonuniform density lambda(x) proportional to x, measured from the pivot at x=0. A 6.0 N force is applied perpendicular to the rod at x=2.0 m. Find the angular acceleration in rad/s^2.",
                "expected_values": [{"label": "angular_acceleration_rad_s2", "value": 2.00, "abs_tol": 0.15, "rel_tol": 0.03}],
                "reference_answer": "For lambda proportional to x, I=(1/2)ML^2=6.0 kg m^2; tau=12 N m; alpha=2.0 rad/s^2",
            },
            {
                "case_id": "limit_series_rlc_power",
                "topic": "limit_ac_circuit",
                "prompt": "A series RLC circuit has R=100 ohm, L=0.50 H, C=10 microF, and is connected to a 120 V rms, 60 Hz source. Find the rms current in A and average power in W.",
                "expected_values": [
                    {"label": "current_A", "value": 0.952, "abs_tol": 0.06, "rel_tol": 0.03},
                    {"label": "power_W", "value": 90.6, "abs_tol": 5.5, "rel_tol": 0.03},
                ],
                "reference_answer": "Z=sqrt(R^2+(omegaL-1/omegaC)^2)=126 ohm; Irms=0.952 A; P=I^2R=90.6 W",
            },
            {
                "case_id": "limit_oppositely_charged_capacitors",
                "topic": "limit_charge_sharing",
                "prompt": "A 4.0 microF capacitor is charged to +12 V and an 8.0 microF capacitor is charged to -6.0 V using the same plate polarity convention. They are disconnected from their batteries and then connected positive plate to positive plate and negative plate to negative plate. Find the final common voltage in V and the energy dissipated in mJ.",
                "expected_values": [
                    {"label": "final_voltage_V", "value": 0.0, "abs_tol": 0.08, "rel_tol": 0.03},
                    {"label": "energy_dissipated_mJ", "value": 0.432, "abs_tol": 0.04, "rel_tol": 0.03},
                ],
                "reference_answer": "Qtotal=4uF*12V+8uF*(-6V)=0, so Vf=0 V; initial energy=0.432 mJ dissipated",
            },
            {
                "case_id": "limit_nonuniform_rod_center_of_mass",
                "topic": "limit_calculus_mechanics",
                "prompt": "A thin rod extends from x=0 to x=3.0 m with linear density lambda(x)=lambda0*(1+x/L), where L=3.0 m. Find the center of mass x-coordinate in meters.",
                "expected_values": [{"label": "x_cm_m", "value": 1.67, "abs_tol": 0.08, "rel_tol": 0.03}],
                "reference_answer": "x_cm=(integral x(1+x/L)dx)/(integral (1+x/L)dx)=5L/9=1.67 m",
            },
            {
                "case_id": "limit_work_energy_stop_before_target",
                "topic": "limit_physical_consistency",
                "prompt": "A 2.0 kg block starts at x=0 moving in the +x direction at 3.0 m/s. A force F(x)=8-4x N acts along x, and a constant 2.0 N kinetic friction force opposes the motion. Does it reach x=5.0 m? If not, find the stopping distance from x=0 in meters.",
                "expected_values": [{"label": "stop_distance_m", "value": 4.10, "abs_tol": 0.18, "rel_tol": 0.03}],
                "reference_answer": "K0+integral_0^s(6-4x)dx=0 gives 9+6s-2s^2=0, so s=4.10 m and it does not reach 5.0 m",
            },
            {
                "case_id": "limit_signed_sphere_field",
                "topic": "limit_gauss_sign",
                "prompt": "A sphere of radius 0.30 m has volume charge density rho(r)=rho0*(1-2r/R), where rho0=1.0 microC/m^3. Using outward radial direction as positive, find the signed radial electric field at r=0.60 m in N/C. Use epsilon0=8.854e-12 F/m.",
                "expected_values": [{"label": "signed_field_N_C", "value": -1.41e3, "abs_tol": 1.1e2, "rel_tol": 0.03}],
                "reference_answer": "Q=4*pi*rho0*R^3*(1/3-1/2)=-2*pi*rho0R^3/3; E=Q/(4*pi*epsilon0*r^2)=-1.41e3 N/C",
            },
            {
                "case_id": "limit_sliding_rod_with_resistor",
                "topic": "limit_motional_emf_dynamics",
                "prompt": "A conducting rod of length 0.30 m moves at 4.0 m/s on rails in a 0.80 T field perpendicular to the circuit. The circuit resistance is 0.60 ohm. Find the induced current in A and the magnetic drag force magnitude on the rod in N.",
                "expected_values": [
                    {"label": "current_A", "value": 1.60, "abs_tol": 0.10, "rel_tol": 0.03},
                    {"label": "drag_force_N", "value": 0.384, "abs_tol": 0.035, "rel_tol": 0.03},
                ],
                "reference_answer": "emf=BLv=0.96 V; I=emf/R=1.60 A; F=ILB=0.384 N",
            },
        ],
        "stuck_base": [
            {
                "case_id": "stuck_known_work_energy_stop_before_target",
                "topic": "stuck_single_path_reachability",
                "prompt": "A 2.0 kg block starts at x=0 moving in the +x direction at 3.0 m/s. A force F(x)=8-4x N acts along x, and a constant 2.0 N kinetic friction force opposes the motion. Does it reach x=5.0 m? If not, find the stopping distance from x=0 in meters. Be careful: a direct calculation at x=5.0 m may be physically invalid if the block stops earlier.",
                "expected_values": [{"label": "stop_distance_m", "value": 4.10, "abs_tol": 0.18, "rel_tol": 0.03}],
                "reference_answer": "K0+integral_0^s(6-4x)dx=0 gives 9+6s-2s^2=0, so s=4.10 m and it does not reach 5.0 m",
            },
            {
                "case_id": "stuck_shifted_turning_point_quadratic_work",
                "topic": "stuck_single_path_reachability",
                "prompt": "A 1.0 kg cart starts at x=0 with speed 2.0 m/s. It moves in +x under F(x)=10-3x N and a constant 1.0 N friction force opposing the motion. Determine whether it reaches x=10.0 m. If it does not, find the first stopping distance from x=0 in meters. You must distinguish a formal negative kinetic energy at the target from the actual earlier turning point.",
                "expected_values": [{"label": "stop_distance_m", "value": 6.21, "abs_tol": 0.25, "rel_tol": 0.03}],
                "reference_answer": "K0=2 J; integral_0^s(9-3x)dx=9s-1.5s^2; 2+9s-1.5s^2=0 gives s=6.21 m, so it does not reach 10 m",
            },
            {
                "case_id": "stuck_particle_potential_barrier_turning_point",
                "topic": "stuck_single_path_reachability",
                "prompt": "A 0.50 kg particle starts at x=0 moving in +x at 4.0 m/s. Its potential energy is U(x)=2x^2-4x joules. No nonconservative forces act. Does it reach x=5.0 m? If not, find the first positive turning-point x-coordinate in meters. Avoid assuming that the particle can pass through a point where K would be negative.",
                "expected_values": [{"label": "turning_point_m", "value": 2.73, "abs_tol": 0.16, "rel_tol": 0.03}],
                "reference_answer": "E=4 J; turning point solves 2x^2-4x=4, so x=1+sqrt(3)=2.73 m and it does not reach 5.0 m",
            },
            {
                "case_id": "stuck_piecewise_force_turning_point",
                "topic": "stuck_single_path_reachability",
                "prompt": "A 1.0 kg object starts at x=0 with speed 5.0 m/s. From x=0 to x=2.0 m, the net force is +3.0 N. For x>2.0 m, the net force is F(x)=7-5x N. Does the object reach x=6.0 m? If not, find the stopping position in meters. The answer requires carrying the kinetic energy across the piecewise boundary before solving for the later root.",
                "expected_values": [{"label": "stop_position_m", "value": 4.32, "abs_tol": 0.20, "rel_tol": 0.03}],
                "reference_answer": "K at x=2 is 12.5+6=18.5 J; for s>2: 18.5+int_2^s(7-5x)dx=0 gives s=4.32 m, so it does not reach 6 m",
            },
            {
                "case_id": "stuck_select_first_root_not_later_root",
                "topic": "stuck_single_path_reachability",
                "prompt": "A 1.0 kg cart has initial kinetic energy 3.0 J at x=0 and moves in +x with net force F(x)=-0.60x^2+3.6x-4.6 N. Does it reach x=8.0 m? If not, find the first stopping distance from x=0. Do not choose a later mathematical root if the kinetic energy already hit zero earlier.",
                "expected_values": [{"label": "first_stop_m", "value": 1.00, "abs_tol": 0.12, "rel_tol": 0.03}],
                "reference_answer": "K(s)=3-4.6s+1.8s^2-0.2s^3=-0.2(s-1)(s-3)(s-5); first positive root is s=1.00 m, so it stops before 8 m",
            },
        ],
        "boundary": [
            {
                "case_id": "boundary_velocity_first_stop_integral",
                "topic": "boundary_first_event_calculus",
                "prompt": "A 1.0 kg cart starts at x=0 with velocity v0=10.0 m/s. Its acceleration is a(t)=-46/3+12t-2t^2 m/s^2 for t>=0. It moves forward only until the first time its velocity becomes zero. Find the stopping distance from x=0 in meters; do not use later mathematical zeros of the velocity.",
                "expected_values": [{"label": "stop_distance_m", "value": 4.17, "abs_tol": 0.18, "rel_tol": 0.03}],
                "reference_answer": "v(t)=10-(46/3)t+6t^2-(2/3)t^3 has first positive zero at t=1.00 s; x_stop=integral_0^1 v(t)dt=25/6=4.17 m",
            },
            {
                "case_id": "boundary_piecewise_energy_late_root",
                "topic": "boundary_piecewise_reachability",
                "prompt": "A 1.0 kg object has initial kinetic energy 18.0 J at x=0. The net force is +4.0 N from x=0 to x=1.0 m, -3.0 N from x=1.0 m to x=3.0 m, and F(x)=2-2x N for x>3.0 m. Does it reach x=6.0 m? If not, find the first stopping position in meters.",
                "expected_values": [{"label": "stop_position_m", "value": 5.47, "abs_tol": 0.22, "rel_tol": 0.03}],
                "reference_answer": "K(1)=22 J, K(3)=16 J, then K(s)=16+integral_3^s(2-2x)dx=19+2s-s^2; first stop is s=1+sqrt(20)=5.47 m before x=6 m",
            },
            {
                "case_id": "boundary_signed_density_total_field",
                "topic": "boundary_signed_gauss_integral",
                "prompt": "A sphere of radius R=0.30 m has volume charge density rho(r)=rho0*(1-3r/R+2(r/R)^2), where rho0=10 microC/m^3. Using outward radial direction as positive, find the signed radial electric field at r=0.60 m in N/C. Use epsilon0=8.854e-12 F/m.",
                "expected_values": [{"label": "signed_field_N_C", "value": -1410.0, "abs_tol": 110.0, "rel_tol": 0.03}],
                "reference_answer": "Q=4*pi*rho0*R^3*(1/3-3/4+2/5)=-4*pi*rho0*R^3/60, so E=Q/(4*pi*epsilon0*r^2)=-1.41e3 N/C",
            },
            {
                "case_id": "boundary_three_capacitor_opposite_polarity",
                "topic": "boundary_charge_sharing_signs",
                "prompt": "Three capacitors are disconnected from all batteries and then connected positive plate to positive plate and negative plate to negative plate. C1=2.0 microF initially has +12 V, C2=3.0 microF initially has -4.0 V using the same plate convention, and C3=5.0 microF is initially uncharged. Find the final common voltage in V and the energy dissipated in mJ.",
                "expected_values": [
                    {"label": "final_voltage_V", "value": 1.20, "abs_tol": 0.08, "rel_tol": 0.03},
                    {"label": "energy_dissipated_mJ", "value": 0.161, "abs_tol": 0.018, "rel_tol": 0.04},
                ],
                "reference_answer": "Q_total=2uF*12V+3uF*(-4V)+5uF*0V=12 uC and C_total=10 uF, so Vf=1.20 V; Ui=0.168 mJ, Uf=0.0072 mJ, dissipated=0.161 mJ",
            },
            {
                "case_id": "boundary_reversed_rc_zero_crossing",
                "topic": "boundary_transient_signs",
                "prompt": "A 200 microF capacitor initially has +6.0 V. At t=0 it is connected through a 3.0 kOhm resistor to an ideal -9.0 V source. Find the first time when the capacitor voltage is zero in seconds and the current magnitude at that instant in mA.",
                "expected_values": [
                    {"label": "zero_crossing_time_s", "value": 0.306, "abs_tol": 0.025, "rel_tol": 0.03},
                    {"label": "current_mA", "value": 3.00, "abs_tol": 0.18, "rel_tol": 0.03},
                ],
                "reference_answer": "V(t)=-9+15exp(-t/0.60); zero crossing t=0.60ln(15/9)=0.306 s and |I|=|-9-0|/3000=3.00 mA",
            },
            {
                "case_id": "boundary_flux_three_factor_product_rule",
                "topic": "boundary_flux_product_rule",
                "prompt": "A rectangular loop has width w(t)=0.20+0.10t^2 m, length ell(t)=0.30+0.050t m, and a perpendicular field B(t)=0.40 exp(-t/2) T. Find the induced emf magnitude at t=2.0 s in mV.",
                "expected_values": [{"label": "emf_mV", "value": 10.3, "abs_tol": 0.8, "rel_tol": 0.04}],
                "reference_answer": "|d(Bw ell)/dt|=|B'w ell+B w' ell+B w ell'| at t=2.0 s =0.0103 V=10.3 mV",
            },
        ],
        "symbolic_tool": [
            {
                "case_id": "symbolic_det_5x5_integer",
                "topic": "symbolic_tool_integer_linear_algebra",
                "prompt": "Compute the determinant of the 5 by 5 matrix [[7,-3,5,2,11],[4,9,-6,1,0],[-2,8,13,-5,3],[10,-1,4,6,-7],[3,12,-9,8,5]]. Return only the determinant as an integer.",
                "expected_values": [{"label": "determinant", "value": 267081, "abs_tol": 0.5, "rel_tol": 0.0}],
                "reference_answer": "determinant = 267081",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_symbolic_solver",
                        "payload": {
                            "operation": "determinant",
                            "matrix": [[7, -3, 5, 2, 11], [4, 9, -6, 1, 0], [-2, 8, 13, -5, 3], [10, -1, 4, 6, -7], [3, 12, -9, 8, 5]],
                        },
                    }
                ],
            },
            {
                "case_id": "symbolic_matrix_power_entry",
                "topic": "symbolic_tool_integer_linear_algebra",
                "prompt": "Let A=[[2,-1,3],[0,4,1],[-2,5,0]]. Compute the row 1, column 3 entry of A^8. Return only that integer entry.",
                "expected_values": [{"label": "matrix_entry", "value": 20405, "abs_tol": 0.5, "rel_tol": 0.0}],
                "reference_answer": "(A^8)_{1,3}=20405",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_symbolic_solver",
                        "payload": {
                            "operation": "matrix_power_entry",
                            "matrix": [[2, -1, 3], [0, 4, 1], [-2, 5, 0]],
                            "power": 8,
                            "row": 1,
                            "col": 3,
                        },
                    }
                ],
            },
            {
                "case_id": "symbolic_recurrence_mod_25",
                "topic": "symbolic_tool_integer_recurrence",
                "prompt": "A sequence is defined by a0=7 and a_{n+1}=(37 a_n + 19) mod 1009. Compute a25. Return only the integer residue.",
                "expected_values": [{"label": "residue", "value": 382, "abs_tol": 0.5, "rel_tol": 0.0}],
                "reference_answer": "a25=382",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_symbolic_solver",
                        "payload": {
                            "operation": "linear_recurrence_mod",
                            "initial": 7,
                            "multiplier": 37,
                            "increment": 19,
                            "modulus": 1009,
                            "steps": 25,
                        },
                    }
                ],
            },
            {
                "case_id": "symbolic_third_derivative_value",
                "topic": "symbolic_tool_calculus",
                "prompt": "Let f(x)=3*x^8-7*x^7+5*x^6-11*x^5+13*x^4-17*x^3+19*x^2-23*x+29. Compute f'''(2). Return only the integer value.",
                "expected_values": [{"label": "third_derivative", "value": 11418, "abs_tol": 0.5, "rel_tol": 0.0}],
                "reference_answer": "f'''(2)=11418",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_symbolic_solver",
                        "payload": {
                            "operation": "derivative_value",
                            "expression": "3*x**8-7*x**7+5*x**6-11*x**5+13*x**4-17*x**3+19*x**2-23*x+29",
                            "variable": "x",
                            "order": 3,
                            "at": 2,
                        },
                    }
                ],
            },
            {
                "case_id": "symbolic_polynomial_coefficient",
                "topic": "symbolic_tool_polynomial_expansion",
                "prompt": "Find the coefficient of x^7 in the expanded polynomial (3*x-2)^4*(2*x+5)^3*(x-7)^2. Return only the integer coefficient.",
                "expected_values": [{"label": "coefficient", "value": -11178, "abs_tol": 0.5, "rel_tol": 0.0}],
                "reference_answer": "coefficient of x^7 is -11178",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_symbolic_solver",
                        "payload": {
                            "operation": "polynomial_coefficient",
                            "expression": "(3*x-2)**4*(2*x+5)**3*(x-7)**2",
                            "variable": "x",
                            "power": 7,
                        },
                    }
                ],
            },
        ],
        "physics_tool": [
            {
                "case_id": "physics_linearized_five_mode_eigen_product",
                "topic": "physics_linearized_modes",
                "prompt": "A driven five-mode oscillator has small-signal dynamics d y/dt = M y, where y contains five mode amplitudes and M is measured in s^-1. Nonreciprocal active couplings make M non-symmetric. M=[[7,-3,5,2,11],[4,9,-6,1,0],[-2,8,13,-5,3],[10,-1,4,6,-7],[3,12,-9,8,5]]. Find the signed product of the five eigenvalues of M, including algebraic multiplicity, in s^-5.",
                "expected_values": [{"label": "eigenvalue_product_s^-5", "value": 267081, "abs_tol": 0.5, "rel_tol": 0.0}],
                "reference_answer": "product of eigenvalues = det(M) = 267081 s^-5",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "linearized_eigenvalue_product",
                            "operator_matrix": [[7, -3, 5, 2, 11], [4, 9, -6, 1, 0], [-2, 8, 13, -5, 3], [10, -1, 4, 6, -7], [3, 12, -9, 8, 5]],
                            "unit": "s^-5",
                        },
                    }
                ],
            },
            {
                "case_id": "physics_dense_first_static_instability",
                "topic": "physics_first_instability_dense_modes",
                "prompt": "A nonconservative five-mode mechanical frame has tangent stiffness K(p)=K0+p*K1, where p is a dimensionless compressive-load factor. Static instability occurs when det(K(p))=0. As p is increased quasistatically from 0, find the first positive instability load factor; do not report later mathematical roots. K0=[[-20,-41,25,41,3],[-53,-32,44,46,3],[-53,-53,58,53,3],[-53,-53,44,67,3],[-25,-25,25,25,8]] and K1=[[0,2,-4,-2,3],[5,3,-10,-2,3],[5,5,-9,-5,3],[5,5,-10,-4,3],[4,4,-4,-4,-1]]. Return only the first positive p.",
                "expected_values": [{"label": "first_instability_load_factor", "value": 1.25, "abs_tol": 0.02, "rel_tol": 0.0}],
                "reference_answer": "det(K(p))=(p-8)(p+14)(2p-21)(4p-5)(5p-33), so the first positive instability is p=1.25",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "first_static_instability",
                            "load_symbol": "p",
                            "stiffness_matrix": [[-20, -41, 25, 41, 3], [-53, -32, 44, 46, 3], [-53, -53, 58, 53, 3], [-53, -53, 44, 67, 3], [-25, -25, 25, 25, 8]],
                            "load_slope_matrix": [[0, 2, -4, -2, 3], [5, 3, -10, -2, 3], [5, 5, -9, -5, 3], [5, 5, -10, -4, 3], [4, 4, -4, -4, -1]],
                            "unit": "load factor",
                        },
                    }
                ],
            },
            {
                "case_id": "physics_thermal_network_hotspot",
                "topic": "physics_steady_thermal_network",
                "prompt": "A five-node anisotropic thermal test plate is reduced to the steady balance A*T=q, where T is the vector of node temperature rises in K above ambient. A=[[12,-2,-1,0,-3],[-2,15,-4,-1,0],[-1,-4,14,-2,-1],[0,-1,-2,11,-3],[-3,0,-1,-3,13]] W/K and q=[480,310,275,190,360] W. Find the temperature rise of node 4 only, in K.",
                "expected_values": [{"label": "node4_temperature_rise_K", "value": 45.6632169667, "abs_tol": 0.03, "rel_tol": 0.0}],
                "reference_answer": "Solving A*T=q gives T4=626545/13721=45.6632169667 K.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "linear_solve_component",
                            "coefficient_matrix": [[12, -2, -1, 0, -3], [-2, 15, -4, -1, 0], [-1, -4, 14, -2, -1], [0, -1, -2, 11, -3], [-3, 0, -1, -3, 13]],
                            "rhs_vector": [480, 310, 275, 190, 360],
                            "component_index": 3,
                            "unit": "K",
                        },
                    }
                ],
            },
            {
                "case_id": "physics_coupled_resonator_lowest_frequency",
                "topic": "physics_generalized_vibration_modes",
                "prompt": "A four-degree-of-freedom coupled resonator has stiffness matrix K=[[120,-35,10,0],[-35,95,-20,15],[10,-20,80,-25],[0,15,-25,70]] N/m and diagonal mass matrix M=diag(3,2,4,5) kg. Natural frequencies satisfy det(K-omega^2*M)=0. Find the lowest positive angular frequency omega in rad/s.",
                "expected_values": [{"label": "lowest_angular_frequency_rad_s", "value": 3.24563758566, "abs_tol": 0.01, "rel_tol": 0.0}],
                "reference_answer": "The smallest positive generalized eigenvalue is omega^2=10.5341633375, so omega=3.24563758566 rad/s.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "lowest_generalized_frequency",
                            "stiffness_matrix": [[120, -35, 10, 0], [-35, 95, -20, 15], [10, -20, 80, -25], [0, 15, -25, 70]],
                            "mass_matrix": [[3, 0, 0, 0], [0, 2, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]],
                            "unit": "rad/s",
                        },
                    }
                ],
            },
            {
                "case_id": "physics_magnetized_cylinder_axis_field",
                "topic": "physics_magnetostatics_finite_magnet",
                "prompt": "A uniformly magnetized finite cylinder has radius 0.9 m, length 2.4 m, and mu0*M=0.8 T. On the symmetry axis outside the magnet, at z=3.7 m from the cylinder center, compute the axial magnetic flux density Bz in T using the finite-cylinder pole-face expression, not a point-dipole approximation.",
                "expected_values": [{"label": "axis_field_T", "value": 0.0170639017453, "abs_tol": 0.0002, "rel_tol": 0.0}],
                "reference_answer": "Bz=0.8/2*((3.7+1.2)/sqrt(0.9^2+4.9^2)-(3.7-1.2)/sqrt(0.9^2+2.5^2))=0.0170639017453 T.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "magnetized_cylinder_axis_field",
                            "mu0_magnetization": "0.8",
                            "radius": "0.9",
                            "length": "2.4",
                            "axial_position": "3.7",
                            "unit": "T",
                        },
                    }
                ],
            },
            {
                "case_id": "physics_beamline_half_trace_stability",
                "topic": "physics_transfer_matrix_stability",
                "prompt": "A paraxial beamline cell is modeled by the ordered product of transfer matrices D(0.7), Q(-0.3), D(1.1), Q(0.2), D(0.8), where D(L)=[[1,L],[0,1]] and Q(k)=[[1,0],[k,1]]. Multiplying in the order listed, compute one half of the trace of the full one-cell transfer matrix. This value determines linear stability; return only trace(Mcell)/2.",
                "expected_values": [{"label": "half_trace", "value": 0.8205, "abs_tol": 0.0005, "rel_tol": 0.0}],
                "reference_answer": "The ordered matrix product gives trace(Mcell)/2=1641/2000=0.8205.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "matrix_product_half_trace",
                            "matrices": [[[1, "7/10"], [0, 1]], [[1, 0], ["-3/10", 1]], [[1, "11/10"], [0, 1]], [[1, 0], ["1/5", 1]], [[1, "4/5"], [0, 1]]],
                        },
                    }
                ],
            },
            {
                "case_id": "physics_relativistic_multi_pion_threshold",
                "topic": "physics_relativistic_threshold_energy",
                "prompt": "A proton beam strikes a stationary proton target. Find the threshold kinetic energy of the beam proton, in MeV, for the reaction p+p -> p+p+pi0+pi+ + pi-. Use rest masses mp=938.272 MeV/c^2, mpi0=134.977 MeV/c^2, and mpi_charged=139.570 MeV/c^2. Use invariant threshold kinematics, not a nonrelativistic estimate.",
                "expected_values": [{"label": "threshold_kinetic_energy_MeV", "value": 919.621619842, "abs_tol": 0.5, "rel_tol": 0.0}],
                "reference_answer": "At threshold sqrt(s)=2*mp+mpi0+2*mpi_charged, so K=((sum_m)^2-mp^2-mp^2)/(2*mp)-mp=919.621619842 MeV.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "relativistic_threshold_kinetic_energy",
                            "projectile_mass": "938.272",
                            "target_mass": "938.272",
                            "final_masses": ["938.272", "938.272", "134.977", "139.570", "139.570"],
                            "unit": "MeV",
                        },
                    }
                ],
            },
        ],
        "fluid_tool": [
            {
                "case_id": "fluid_rough_pipe_colebrook_drop",
                "topic": "fluid_turbulent_pipe_friction",
                "prompt": "Water at rho=998 kg/m^3 and dynamic viscosity 1.002e-3 Pa*s flows through an 85 m long commercial pipe of diameter 0.12 m and absolute roughness 0.00015 m. The volume flow rate is 0.018 m^3/s. Using the Colebrook-White equation for the Darcy friction factor, compute the Darcy friction factor and the pressure drop in Pa. Do not use a smooth-pipe approximation.",
                "expected_values": [
                    {"label": "darcy_friction_factor", "value": 0.022015544236, "abs_tol": 0.0002, "rel_tol": 0.0},
                    {"label": "pressure_drop_Pa", "value": 19710.9662578, "abs_tol": 80.0, "rel_tol": 0.0},
                ],
                "reference_answer": "Colebrook-White gives f=0.022015544236 and Delta p=f*(L/D)*rho*V^2/2=19710.9662578 Pa.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "rough_pipe_pressure_drop",
                            "density": 998,
                            "dynamic_viscosity": 0.001002,
                            "diameter": 0.12,
                            "length": 85,
                            "roughness": 0.00015,
                            "flow_rate": 0.018,
                        },
                    }
                ],
            },
            {
                "case_id": "fluid_supersonic_nozzle_area_mach",
                "topic": "fluid_compressible_nozzle_flow",
                "prompt": "Air expands isentropically in a converging-diverging nozzle with gamma=1.4. At a station in the diverging section, A/A*=2.8 and the flow is on the supersonic branch. Compute the local Mach number and the static-to-stagnation pressure ratio p/p0.",
                "expected_values": [
                    {"label": "mach_number", "value": 2.56416809718, "abs_tol": 0.01, "rel_tol": 0.0},
                    {"label": "pressure_ratio", "value": 0.0529757394832, "abs_tol": 0.001, "rel_tol": 0.0},
                ],
                "reference_answer": "Solving the area-Mach relation on the supersonic branch gives M=2.56416809718 and p/p0=0.0529757394832.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "isentropic_nozzle_supersonic",
                            "gamma": 1.4,
                            "area_ratio": 2.8,
                        },
                    }
                ],
            },
            {
                "case_id": "fluid_rectangular_hydraulic_jump",
                "topic": "fluid_open_channel_jump",
                "prompt": "A rectangular channel has upstream depth y1=0.45 m and upstream mean velocity V1=7.2 m/s just before a hydraulic jump. Use g=9.81 m/s^2. Compute the sequent downstream depth y2 and the specific energy loss across the jump, both in meters.",
                "expected_values": [
                    {"label": "downstream_depth_m", "value": 1.96739328195, "abs_tol": 0.02, "rel_tol": 0.0},
                    {"label": "energy_loss_m", "value": 0.986576534576, "abs_tol": 0.02, "rel_tol": 0.0},
                ],
                "reference_answer": "Fr1=3.426823495, y2=y1/2*(sqrt(1+8*Fr1^2)-1)=1.96739328195 m, and energy loss=(y2-y1)^3/(4*y1*y2)=0.986576534576 m.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "hydraulic_jump_rectangular",
                            "upstream_depth": 0.45,
                            "upstream_velocity": 7.2,
                            "gravity": 9.81,
                        },
                    }
                ],
            },
            {
                "case_id": "fluid_laminar_flat_plate_drag",
                "topic": "fluid_boundary_layer_drag",
                "prompt": "Air with rho=1.20 kg/m^3 and dynamic viscosity 1.8e-5 Pa*s flows at 22 m/s over a smooth flat plate of length 1.5 m and width 0.8 m. Assuming the boundary layer remains laminar over the full plate, compute the average skin-friction coefficient and the one-sided drag force in N.",
                "expected_values": [
                    {"label": "average_skin_friction_coefficient", "value": 0.000895337417351, "abs_tol": 0.00002, "rel_tol": 0.0},
                    {"label": "drag_force_N", "value": 0.312007183199, "abs_tol": 0.02, "rel_tol": 0.0},
                ],
                "reference_answer": "Re_L=2.2e6, Cf_avg=1.328/sqrt(Re_L)=0.000895337417351, and D=0.5*rho*U^2*L*b*Cf_avg=0.312007183199 N.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "laminar_flat_plate_drag",
                            "density": 1.20,
                            "dynamic_viscosity": 0.000018,
                            "velocity": 22,
                            "length": 1.5,
                            "width": 0.8,
                        },
                    }
                ],
            },
            {
                "case_id": "fluid_capillary_gravity_wave_speed",
                "topic": "fluid_surface_wave_dispersion",
                "prompt": "A water surface wave has wavelength 0.045 m in water depth 0.18 m. Use rho=1000 kg/m^3, surface tension sigma=0.072 N/m, and g=9.81 m/s^2. Using the capillary-gravity dispersion relation omega^2=(g*k+sigma*k^3/rho)*tanh(k*h), compute the phase speed in m/s.",
                "expected_values": [{"label": "phase_speed_m_s", "value": 0.283393800425, "abs_tol": 0.003, "rel_tol": 0.0}],
                "reference_answer": "k=2*pi/0.045 and c=omega/k=0.283393800425 m/s.",
                "tool_invocations": [
                    {
                        "skill_name": "benchmark_physics_solver",
                        "payload": {
                            "operation": "capillary_gravity_phase_speed",
                            "density": 1000,
                            "surface_tension": 0.072,
                            "depth": 0.18,
                            "wavelength": 0.045,
                            "gravity": 9.81,
                        },
                    }
                ],
            },
        ],
    },
}


def _copy_direct_9b_case(case: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(case)
    copied["expected_values"] = [dict(item) for item in copied.get("expected_values", [])]
    copied["tool_invocations"] = [
        {
            "skill_name": str(item.get("skill_name", "")),
            "payload": dict(item.get("payload", {})) if isinstance(item.get("payload"), dict) else {},
        }
        for item in copied.get("tool_invocations", [])
        if isinstance(item, dict)
    ]
    return copied


def direct_9b_benchmark_payload(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return direct model benchmark suites and prompts for external harnesses."""

    del params
    base_cases = DIRECT_9B_BENCHMARK_DATA["cases"]
    em_cases = [_copy_direct_9b_case(item) for item in base_cases["em"]]
    trap_cases = [_copy_direct_9b_case(item) for item in base_cases["traps"]]
    freeform_cases = [_copy_direct_9b_case(item) for item in base_cases["freeform"]]
    stuck_base = [_copy_direct_9b_case(item) for item in base_cases["stuck_base"]]
    boundary_cases = [_copy_direct_9b_case(item) for item in base_cases["boundary"]]
    symbolic_tool_cases = [_copy_direct_9b_case(item) for item in base_cases["symbolic_tool"]]
    physics_tool_cases = [_copy_direct_9b_case(item) for item in base_cases["physics_tool"]]
    fluid_tool_cases = [_copy_direct_9b_case(item) for item in base_cases["fluid_tool"]]
    suites = {
        "ap1": [_copy_direct_9b_case(item) for item in base_cases["ap1"]],
        "em": em_cases,
        "traps": trap_cases,
        "freeform": freeform_cases,
        "hard": [*em_cases, *trap_cases, *freeform_cases],
        "limit": [_copy_direct_9b_case(item) for item in base_cases["limit"]],
        "stuck": [*stuck_base[1:], stuck_base[0]],
        "boundary": boundary_cases,
        "symbolic_tool": symbolic_tool_cases,
        "physics_tool": physics_tool_cases,
        "fluid_tool": fluid_tool_cases,
    }
    return {
        "default_suite": DIRECT_9B_BENCHMARK_DATA["default_suite"],
        "system_prompt": DIRECT_9B_BENCHMARK_DATA["system_prompt"],
        "freeform_system_prompt": DIRECT_9B_BENCHMARK_DATA["freeform_system_prompt"],
        "suites": suites,
    }

