use std::env;

use anyhow::{Error, Result};
use yolo_tir::BoxInfo;
use std::env;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};



struct YoloTirInfer {
    num_messages: AtomicU32,
    img_msg: sensor_msgs::msg::Image,
    node: Arc<rclrs::Node>,
    subscriber: Mutex<Option<Arc<rclrs::Subscription<sensor_msgs::msg::Image>>>>,
    publisher: rclrs::Publisher<vision_msgs::msg::Detection2DArray>
}

impl YoloTirInfer {
    pub fn new(name: &str, sub_topic: &str, pub_topic: &str) -> Result<Arc<Self>, rclrs::RclrsError> {
        // create node in C-style
        let context = rclrs::Context::new(env::args())?;

        let node = rclrs::create_node(&context, name)?;

        let tir_obj = Arc::new(YoloTirInfer {
            num_messages: Default::default(),
            img_msg: (),
            node,
            subscriber: None.into(),
            publisher: None.into()
        });

        let publisher = tir_obj.node.create_publisher::<std_msgs::msg::String>(pub_topic, rclrs::QOS_PROFILE_DEFAULT)?;
        // clone to read the tir object inside the lambda function without taking ownership
        let obj_aux = Arc::clone(&tir_obj);
        tir_obj.node.create_subscription::<sensor_msgs::msg::Image, _>(
                sub_topic,
                rclrs::QOS_PROFILE_DEFAULT,
                move |msg: std_msgs::msg::String| {
                    obj_aux.callback(msg);
                },
            )?;



        Ok(tir_obj)
    }

    fn callback(&self, img_msg: sensor_msgs::msg::Image) {
        // todo infer image
        let hyp_ps = vision_msgs::msg::ObjectHypothesisWithPose{
            hypothesis: vision_msgs::msg::ObjectHypothesis{
                class_id: "".to_string(),
                score: 0.0,
            },
            pose: Default::default()
        };
        let msg = vision_msgs::msg::Detection2D{
            header: Default::default(),
            results: vec![hyp_ps],
            bbox: Default::default(),
            id: "".to_string(),
        };
        self.publisher.publish(&msg)?
    }
}

fn main() -> Result<(), Error> {

    let  yolo_node = YoloTirInfer::new("yolo_tir_node", "/tir/image", "/tir/detection")?;

    let executor = rclrs::SingleThreadedExecutor::new();
    executor.add_node(&yolo_node.node)?;

    executor.spin().map_err(|err| err.into())
}
