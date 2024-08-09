use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    StreamConfig,
};
use crossbeam_channel::Receiver;
use eframe::egui;
use fundsp::hacker::*;

#[derive(Clone)]
pub struct InputNode {
    receiver: Receiver<f32>,
    tick: Instant,
}

impl InputNode {
    pub fn new(receiver: Receiver<f32>) -> Self {
        InputNode {
            receiver,
            tick: Instant::now(),
        }
    }
}

// #[derive(Clone)]
// pub struct PitchShift {
//     resampler: Arc<SincFixedOut<f64>>
// }

// impl AudioNode for PitchShift {
//     const ID: u64 = 1239;

//     type Inputs = U1;

//     type Outputs = U1;

//     fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
//         todo!()
//     }
// }

impl AudioNode for InputNode {
    const ID: u64 = 87;
    type Inputs = U0;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, _input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let value = self.receiver.try_recv().unwrap_or(0.0);
        if self.tick.elapsed() > Duration::from_secs(10) {
            println!("Sound queue length {}", self.receiver.len());
            self.tick = Instant::now();
        }
        [value].into()
    }
}

enum AppState {
    Init,
    Selecting {
        record: Vec<cpal::Device>,
        playback: Vec<cpal::Device>,
        selected_rec: Option<usize>,
        selected_pb: Option<usize>,
    },
    Working(cpal::Stream, cpal::Stream),
}

struct App {
    state: AppState,
}

impl Default for App {
    fn default() -> Self {
        Self {
            state: AppState::Init,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| match &mut self.state {
            AppState::Init => {
                ui.label("loading");
                let host = cpal::default_host();
                self.state = AppState::Selecting {
                    record: host.input_devices().unwrap().collect(),
                    playback: host.output_devices().unwrap().collect(),
                    selected_rec: None,
                    selected_pb: None,
                }
            }
            AppState::Selecting {
                record,
                playback,
                selected_rec,
                selected_pb,
            } => {
                let size = ui.available_size();
                ui.horizontal(|ui| {
                    ui.set_max_height(size.x);

                    ui.vertical(|ui| {
                        ui.label("Recording");
                        egui::ScrollArea::vertical()
                            .id_source("rec")
                            .show(ui, |ui| {
                                for (i, d) in record.iter().enumerate() {
                                    if ui
                                        .selectable_label(
                                            Some(i) == *selected_rec,
                                            d.name().unwrap_or("Unknown".to_string()),
                                        )
                                        .clicked()
                                    {
                                        *selected_rec = Some(i);
                                    }
                                }
                            });
                    });
                    ui.vertical(|ui| {
                        ui.label("Playback");
                        egui::ScrollArea::vertical().id_source("pb").show(ui, |ui| {
                            for (i, d) in playback.iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        Some(i) == *selected_pb,
                                        d.name().unwrap_or("Unknown".to_string()),
                                    )
                                    .clicked()
                                {
                                    *selected_pb = Some(i);
                                }
                            }
                        });
                    });
                });
                if let (Some(sel_rec), Some(sel_pb)) = (selected_rec, selected_pb) {
                    let (s1, s2) = start_streams(&record[*sel_rec], &playback[*sel_pb]);
                    self.state = AppState::Working(s1, s2);
                }
            }
            AppState::Working(_s1, _s2) => {
                if ui.button("Reset").clicked() {
                    self.state = AppState::Init;
                }
            }
        });
    }
}

fn start_streams(
    rec_device: &cpal::Device,
    pb_device: &cpal::Device,
) -> (cpal::Stream, cpal::Stream) {
    let (sound_sender, sound_receiver) = crossbeam_channel::bounded(4096 * 2);
    let rec_stream = {
        let config: StreamConfig = rec_device
            .default_input_config()
            .expect("Failed to get default input config")
            .into();
        println!("Record config on {} {config:?}", rec_device.name().unwrap());

        let channels = config.channels;

        let stream = rec_device
            .build_input_stream(
                &config,
                move |data: &[f32], _| {
                    if data.len() <= 2 {
                        return;
                    }
                    // average values together
                    for v in data.chunks(channels as usize) {
                        _ = sound_sender.try_send(v.iter().sum());
                    }
                },
                move |err| {
                    eprintln!("an error occurred on stream: {}", err);
                },
                None,
            )
            .unwrap();

        stream.play().unwrap();
        stream
    };
    let playback_stream = {
        let config: StreamConfig = pb_device.default_output_config().unwrap().into();
        let channels = config.channels as usize;
        println!(
            "Playback config on {} {config:?}",
            pb_device.name().unwrap()
        );

        // FIXME: resample

        let mut graph = {
            let input = {
                // let wav = hound::WavReader::open("comb.wav")
                //     .unwrap()
                //     .samples()
                //     .map(|s| s.unwrap())
                //     .step_by(2)
                //     .collect::<Vec<f32>>();
                // wavech(&std::sync::Arc::new(Wave::from_samples(44100.0, &wav)), 0, None)
                An(InputNode::new(sound_receiver))
            };
            let shift_bins = 3;
            let pitch_shift = resynth(1024 * 2, move |fft: &mut FftWindow| {
                for i in 1..(fft.bins() - shift_bins - 1) {
                    fft.set(0, i, fft.at(0, i + shift_bins));
                }
            });

            // let knee = 0.1;
            // let pow = 0.84;
            // let shaper = shape_fn(move |v| {
            //     if v.abs() < knee {
            //         v
            //     } else {
            //         v.abs().powf(pow) * v.signum() / knee.powf(pow) * knee
            //     }
            // });

            let compress = mul(5.0) >> limiter(0.002, 0.002);
            let q = 2.0;
            let initial_filtering =
                pitch_shift >> highpass_hz(400.0, q) >> lowpass_hz(3000.0, q) >> compress.clone();

            let lower = phaser(0.4, |t| fundsp::hacker::sin_hz(3.0, t) * 0.3 + 0.3)
                >> lowpass_hz(700.0, q)
                >> compress.clone();

            let higher = highpass_hz(1000.0, q) >> compress.clone();

            let graph = input >> initial_filtering >> split::<U2>() >> (lower | higher) >> join();

            let mut graph = BlockRateAdapter::new(Box::new(graph));
            graph.set_sample_rate(config.sample_rate.0 as f64);
            graph
        };

        let stream = pb_device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    for i in 0..data.len() / channels {
                        let sample = graph.get_mono();
                        for c in 0..channels {
                            data[i * channels + c] = sample;
                        }
                    }
                },
                move |err| {
                    eprintln!("an error occurred on stream: {}", err);
                },
                None,
            )
            .unwrap();
        stream.play().unwrap();
        stream
    };
    (playback_stream, rec_stream)
}

#[allow(clippy::precedence)]
fn main() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 300.0])
            .with_min_inner_size([600.0, 300.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Combiner",
        native_options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
    .unwrap();
}
