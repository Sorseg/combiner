use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait};
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
    let device = host.default_input_device().unwrap();

    let config = device
        .default_input_config()
        .expect("Failed to get default input config");

    let (sound_sender, sound_receiver) = crossbeam_channel::unbounded();

    let _stream = device
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

    let playback_device = host.default_output_device().unwrap();
    let config = playback_device.default_output_config().unwrap();

    let input = An(InputNode {
        receiver: sound_receiver,
    });

    let filter = resonator_hz(1700.0, 1300.0);

    let graph = input >> filter;
    let mut graph = BlockRateAdapter::new(Box::new(graph));
    graph.set_sample_rate(config.sample_rate().0 as f64);

    let _stream = device
        .build_output_stream(
            &config.into(),
            move |data: &mut [f32], _| {
                for e in data {
                    *e = graph.get_mono();
                }
            },
            move |err| {
                eprintln!("an error occurred on stream: {}", err);
            },
            None,
        )
        .unwrap();

    loop {
        std::thread::sleep(Duration::from_secs(1))
    }
}
