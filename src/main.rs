// Import necessary standard library and external crate modules
use std::collections::HashMap;
use plotters::prelude::*;
use std::thread;
use std::time::Duration;

/// The `Component` trait defines the common interface for all components in the circuit,
/// ensuring they can interact with the `Circuit` and perform operations like voltage
/// drop calculation and temperature updates.
trait Component {
    /// Get the resistance of the component.
    /// This may vary depending on the component's state, e.g., temperature.
    fn get_resistance(&self) -> f64;

    /// Get the current temperature of the component.
    fn get_temperature(&self) -> f64;

    /// Calculate the voltage drop across the component given the current.
    fn calculate_voltage_drop(&self, current: f64) -> f64;

    /// Set the upstream voltage for the component (used for propagating voltage in the circuit).
    fn set_upstream_voltage(&mut self, voltage: f64);

    /// Get the upstream voltage for the component.
    fn get_upstream_voltage(&self) -> f64;

    /// Update the component's temperature based on the current and elapsed time.
    fn update_temperature(&mut self, time_step: f64, current: f64);
}

/// The `Circuit` struct models an electrical circuit, which contains components (such as resistors)
/// and the connections between them. It manages the simulation of the circuit over time.
struct Circuit {
    /// A vector of components in the circuit. Each component is wrapped in a `Box` to allow for
    /// dynamic dispatch via the `Component` trait.
    components: Vec<Box<dyn Component>>,

    /// A HashMap storing the connections between components. Each component has a list of indices
    /// to other components it is connected to.
    connections: HashMap<usize, Vec<usize>>,

    /// The current flowing through the circuit.
    current: f64,

    /// A vector to store the time data for plotting.
    time_data: Vec<f64>,

    /// A vector to store the current data over time for plotting.
    current_data: Vec<f64>,

    /// A vector to store the temperature data of each component over time.
    temperature_data: Vec<Vec<f64>>,
}

impl Circuit {
    /// Creates a new `Circuit` instance, initializing empty data structures.
    fn new() -> Self {
        Self {
            components: Vec::new(),
            connections: HashMap::new(),
            current: 0.0,
            time_data: Vec::new(),
            current_data: Vec::new(),
            temperature_data: Vec::new(),
        }
    }

    /// Adds a new component to the circuit, returning its index in the components vector.
    ///
    /// # Parameters
    /// - `component`: The component to add to the circuit.
    ///
    /// # Returns
    /// The index of the added component.
    fn add_component(&mut self, component: Box<dyn Component>) -> usize {
        self.components.push(component);
        self.temperature_data.push(Vec::new());
        self.components.len() - 1
    }

    /// Connects two components in the circuit.
    ///
    /// # Parameters
    /// - `from`: The index of the component that is sending the current.
    /// - `to`: The index of the component that is receiving the current.
    fn connect(&mut self, from: usize, to: usize) {
        self.connections.entry(from).or_default().push(to);
    }

    /// Simulates the behavior of the circuit over a given period, updating the components
    /// and collecting data for plotting.
    ///
    /// # Parameters
    /// - `total_time`: The total simulation time in seconds.
    /// - `simulation_speed`: The speed of the simulation, determining the time step in milliseconds.
    fn simulate(&mut self, total_time: f64, simulation_speed: f64) {
        let mut time_elapsed = 0.0;
        let time_step = simulation_speed / 1000.0; // Conversion to seconds

        println!("Starting circuit simulation...");

        while time_elapsed <= total_time {
            // Calculate total resistance in the circuit by summing the resistances of all components.
            let total_resistance: f64 = self
                .components
                .iter()
                .map(|component| component.get_resistance())
                .sum();

            // Calculate the current using Ohm's law (I = V/R), assuming the first component is the power supply.
            self.current = if total_resistance > 0.0 {
                self.components[0].get_upstream_voltage() / total_resistance
            } else {
                0.0
            };

            let mut updates: Vec<(usize, f64)> = Vec::new();

            // Iterate over each component to calculate voltage drop and propagate voltage updates.
            for (i, component) in self.components.iter_mut().enumerate() {
                let voltage_drop = component.calculate_voltage_drop(self.current);
                if let Some(neighbors) = self.connections.get(&i) {
                    for &neighbor_idx in neighbors {
                        updates.push((neighbor_idx, voltage_drop));
                    }
                }
            }

            // Apply the calculated voltage drop to all connected components.
            for (neighbor_idx, voltage_drop) in updates {
                if let Some(component) = self.components.get_mut(neighbor_idx) {
                    component.set_upstream_voltage(voltage_drop);
                }
            }

            // Update the temperature of each component based on the current and time step.
            for (i, component) in self.components.iter_mut().enumerate() {
                component.update_temperature(time_step, self.current);
                self.temperature_data[i].push(component.get_temperature());
            }

            // Record time and current for plotting purposes.
            self.time_data.push(time_elapsed);
            self.current_data.push(self.current);

            // Uncomment for optional print output at each time step
            // println!("\nTime Elapsed: {:.2} seconds", time_elapsed);
            // println!("Circuit Current: {:.3} A", self.current);
            // for (i, component) in self.components.iter().enumerate() {
            //     println!(
            //         "Component {}: Resistance: {:.2} Ohms, Voltage Drop: {:.2} V, Temperature: {:.2} °C",
            //         i,
            //         component.get_resistance(),
            //         component.calculate_voltage_drop(self.current),
            //         component.get_temperature()
            //     );
            // }
            // thread::sleep(Duration::from_millis(simulation_speed as u32)); 

            time_elapsed += time_step;
        }

        println!("Simulation complete!");
        self.plot_results();
    }

    /// Generates and saves a plot of the simulation results, showing both current and component temperatures.
    fn plot_results(&self) {
        // Create a new drawing area for the plot, specifying the output file and resolution.
        let root = BitMapBackend::new("simulation_results.png", (1920, 1080))
        .into_drawing_area();
        root.fill(&WHITE).unwrap();

        // Split the drawing area into two parts: one for current, one for temperature.
        let (upper, lower) = root.split_vertically(360);

        // Create the current chart on the upper part of the plot.
        let mut current_chart = ChartBuilder::on(&upper)
            .caption("Circuit Current Over Time", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..self.time_data.last().copied().unwrap_or(1.0), 
                                0.0..1.2 * self.current_data.iter().cloned().fold(0.0, f64::max))
            .unwrap();

        current_chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Current (A)")
            .draw()
            .unwrap();

        current_chart
            .draw_series(LineSeries::new(
                self.time_data.iter().zip(self.current_data.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))
            .unwrap()
            .label("Current")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        current_chart.configure_series_labels().draw().unwrap();

        // Find the maximum temperature across all components to set the upper limit for the temperature plot.
        let max_temperature = self.temperature_data.iter()
            .flat_map(|temp_data| temp_data.iter())
            .cloned()
            .fold(0.0, f64::max);

        // Create the temperature chart on the lower part of the plot.
        let mut temp_chart = ChartBuilder::on(&lower)
            .caption("Component Temperatures Over Time", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0.0..self.time_data.last().copied().unwrap_or(1.0),
                0.0..1.2 * max_temperature,  // Scale y-axis based on the highest temperature
            )
            .unwrap();

        temp_chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Temperature (°C)")
            .draw()
            .unwrap();

        // Plot the temperature data for each component.
        for (i, temp_data) in self.temperature_data.iter().enumerate() {
            temp_chart
                .draw_series(LineSeries::new(
                    self.time_data.iter().zip(temp_data.iter()).map(|(&x, &y)| (x, y)),
                    &Palette99::pick(i),
                ))
                .unwrap()
                .label(format!("Component {}", i))
                .legend(move |(x, y)| {
                    Circle::new((x, y), 5, Palette99::pick(i).filled())
                });
        }

        temp_chart.configure_series_labels().draw().unwrap();

        println!("Results plotted and saved to simulation_results.png");
    }
}

/// The `Resistor` struct represents a resistor component in the circuit.
struct Resistor {
    base_resistance: f64,
    temperature: f64,
    resistance_coefficient: f64,
    upstream_voltage: f64,
    mass: f64,
    specific_heat_capacity: f64,
    heat_transfer_coefficient: f64,
}

impl Component for Resistor {
    fn get_resistance(&self) -> f64 {
        self.base_resistance * (1.0 + self.resistance_coefficient * (self.temperature - 20.0))
    }

    fn get_temperature(&self) -> f64 {
        self.temperature
    }

    fn calculate_voltage_drop(&self, current: f64) -> f64 {
        current * self.get_resistance()
    }

    fn set_upstream_voltage(&mut self, voltage: f64) {
        self.upstream_voltage = voltage;
    }

    fn get_upstream_voltage(&self) -> f64 {
        self.upstream_voltage
    }

    fn update_temperature(&mut self, time_step: f64, current: f64) {
        // Calculate the power dissipated in the resistor using P = I^2 * R.
        let energy_in  = self.get_resistance() * current.powi(2) * time_step; // P = I^2 * R
        // Calculate the temperature increase due to heat generated by current flow.
        let heat_gain: f64 = energy_in / (self.mass * self.specific_heat_capacity);
        // Calculate the temperature decrease due to heat loss to the environment.
        let heat_loss = self.heat_transfer_coefficient * self.temperature * time_step;

        // Update the temperature of the resistor.
        self.temperature += heat_gain - heat_loss;
    }
}

/// The `PowerSupply` struct represents a power supply component in the circuit.
struct PowerSupply {
    base_resistance: f64,
    temperature: f64,
    voltage: f64,
}

impl Component for PowerSupply {
    fn get_resistance(&self) -> f64 {
        self.base_resistance
    }

    fn get_temperature(&self) -> f64 {
        self.temperature
    }

    fn calculate_voltage_drop(&self, _: f64) -> f64 {
        self.voltage
    }

    fn set_upstream_voltage(&mut self, _: f64) {}

    fn get_upstream_voltage(&self) -> f64 {
        self.voltage
    }

    fn update_temperature(&mut self, _: f64, _: f64) {}
}

fn main() {
    // Create a new circuit instance.
    let mut circuit = Circuit::new();

    // Define a power supply and two resistors (representing light bulbs) as components.
    let power_supply = Box::new(PowerSupply {
        base_resistance: 0.1,
        temperature: 25.0,
        voltage: 10.0,
    });

    let bulb_1 = Box::new(Resistor {
        base_resistance: 10.0,
        temperature: 50.0,
        resistance_coefficient: 0.004,
        upstream_voltage: 0.0,
        mass: 0.01,
        specific_heat_capacity: 900.0,
        heat_transfer_coefficient: 0.1,
    });

    let bulb_2 = Box::new(Resistor {
        base_resistance: 10.0,
        temperature: 100.0,
        resistance_coefficient: 0.04,
        upstream_voltage: 0.0,
        mass: 0.01,
        specific_heat_capacity: 900.0,
        heat_transfer_coefficient: 0.1,
    });

    // Add the components to the circuit and store their indices.
    let ps_idx = circuit.add_component(power_supply);
    let r1_idx = circuit.add_component(bulb_1);
    let r2_idx = circuit.add_component(bulb_2);

    // Connect the components: power supply -> bulb 1 -> bulb 2.
    circuit.connect(ps_idx, r1_idx);
    circuit.connect(r1_idx, r2_idx);

    // Start the simulation.
    circuit.simulate(100.0, 1000.0);
}