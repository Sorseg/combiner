use std::time::{Duration, Instant};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    StreamConfig,
};
use crossbeam_channel::Receiver;
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

pub struct PitchShift {}

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

#[allow(clippy::precedence)]
fn main() {
    let report_dur = Duration::from_secs(4);
    let host = cpal::default_host();
    let (sound_sender, sound_receiver) = crossbeam_channel::bounded(4096 * 2);
    let _rec_stream = {
        let device = host.default_input_device().unwrap();

        let config: StreamConfig = device
            .default_input_config()
            .expect("Failed to get default input config")
            .into();
        let channels = config.channels;

        println!("Record config on {} {config:?}", device.name().unwrap());
        let mut read_count = 0;
        let mut start = Instant::now();

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _| {
                    // take only values from the first channel
                    for v in data.chunks(channels as usize) {
                        read_count += 1;
                        if start.elapsed() > report_dur {
                            println!(
                                "Recorded {} samples",
                                read_count as f32 / report_dur.as_secs_f32()
                            );
                            start = Instant::now();
                            read_count = 0;
                        }
                        _ = sound_sender.try_send(v[0] + v[1]);
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
    let _playback_stream = {
        let playback_device = host.default_output_device().unwrap();
        let config: StreamConfig = playback_device.default_output_config().unwrap().into();
        let channels = config.channels as usize;
        println!(
            "Playback config on {} {config:?}",
            playback_device.name().unwrap()
        );

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

            let pitch_shift = resynth(1024 * 2, |fft: &mut FftWindow| {
                for i in 1..(fft.bins() - 4) {
                    fft.set(0, i, fft.at(0, i + 3));
                }
            });

            let knee = 0.1;
            let pow = 0.84;
            let shaper = shape_fn(move |v| {
                if v.abs() < knee {
                    v
                } else {
                    v.abs().powf(pow) * v.signum() / knee.powf(pow) * knee
                }
            });

            let compress = shaper >> mul(5.0) >> limiter(0.002, 0.002);
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

        let mut read_samples = 0;
        let mut started = Instant::now();

        let stream = playback_device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    for i in 0..data.len() / channels {
                        read_samples += 1;
                        let sample = graph.get_mono();
                        for c in 0..channels {
                            data[i * channels + c] = sample;
                        }
                        if started.elapsed() > report_dur {
                            println!(
                                "Read {} samples from the graph",
                                read_samples as f32 / report_dur.as_secs_f32()
                            );
                            read_samples = 0;
                            started = Instant::now();
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
    loop {
        std::thread::sleep(Duration::from_secs(1))
    }
}
