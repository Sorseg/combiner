use std::time::Duration;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    StreamConfig,
};
use crossbeam_channel::Receiver;
use fundsp::hacker::{resonator_hz, An, AudioNode, AudioUnit, BlockRateAdapter, Frame, U0, U1};

#[derive(Clone)]
pub struct InputNode {
    receiver: Receiver<f32>,
}

impl InputNode {
    pub fn new(receiver: Receiver<f32>) -> Self {
        InputNode { receiver }
    }
}

impl AudioNode for InputNode {
    const ID: u64 = 87;
    type Inputs = U0;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, _input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let value = self.receiver.try_recv().unwrap_or(0.0);
        [value].into()
    }
}

fn main() {
    let host = cpal::default_host();
    let (sound_sender, sound_receiver) = crossbeam_channel::bounded(4096);
    let _rec_stream = {
        let device = host.default_input_device().unwrap();

        let config = device
            .default_input_config()
            .expect("Failed to get default input config");

        println!("Record config on {} {config:?}", device.name().unwrap());

        let stream = device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    for v in data {
                        sound_sender.send(*v).unwrap();
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
        println!(
            "Playback config on {} {config:?}",
            playback_device.name().unwrap()
        );

        let input = An(InputNode {
            receiver: sound_receiver,
        });

        let filter = resonator_hz(1700.0, 1300.0);

        let graph = input >> filter;
        let mut graph = BlockRateAdapter::new(Box::new(graph));
        graph.set_sample_rate(config.sample_rate.0 as f64);

        let stream = playback_device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    for i in 0..data.len() / 2 {
                        data[i * 2] = graph.get_mono();
                        data[i * 2 + 1] = graph.get_mono();
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
