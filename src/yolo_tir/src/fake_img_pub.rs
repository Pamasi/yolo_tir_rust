use anyhow::{Error,Result};

use std::env;
use std::sync::Arc;
use image;
use builtin_interfaces;

use std_msgs::msg::Header as HeaderMsg;
use sensor_msgs::msg::Image as ImageMsg;


fn main() -> Result<(), Error> {
    let context = rclrs::Context::new(env::args())?;

    let node = rclrs::create_node(&context, "fake_image_publisher")?;
    let empty_str_default: Arc<str> = Arc::from("");

    node.use_undeclared_parameters().set("image_topic", empty_str_default.clone()).unwrap();
    node.use_undeclared_parameters().set("image_path", empty_str_default.clone()).unwrap();
    let img_topic = node.use_undeclared_parameters().get::<Arc<str>>("image_topic").unwrap();

    let publisher =
        node.create_publisher::<ImageMsg>(&img_topic[..], rclrs::QOS_PROFILE_DEFAULT)?;

    let mut message = std_msgs::msg::String::default();

    let mut frame_id: u32 = 1;

    let  NANO_CONV = 10^9;

    while context.ok() {
        let img_path = node.use_undeclared_parameters().get::<Arc<str>>("image_path").unwrap();
        let img_buf = image::open(&img_path[..])?.grayscale();
        let nsec = node.get_clock().now().nsec;

        let header = HeaderMsg{
            stamp: builtin_interfaces::msg::Time{
                sec: (nsec*NANO_CONV) as i32,
                nanosec: nsec as u32
            },
            frame_id: frame_id.to_string()
        };

        let height = img_buf.height() as u32;
        let width = img_buf.width() as u32;
        let encoding = "mono8".to_string();
        let step = height*width;
        let data = img_buf.as_bytes().to_vec();

        let message = ImageMsg{
            header,
            height,
            width,
            encoding,
            is_bigendian: 1,
            step,
            data

        };

        publisher.publish(&message)?;
        frame_id += 1;
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    Ok(())
}