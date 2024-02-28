use anyhow::{Error, Result};
use builtin_interfaces;
use image;
use std::env;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use sensor_msgs::msg::Image as ImageMsg;
use std_msgs::msg::Header as HeaderMsg;

fn main() -> Result<(), Error> {
    let context = rclrs::Context::new(env::args())?;

    let node = rclrs::create_node(&context, "fake_image_publisher")?;
    let empty_str_default: Arc<str> = Arc::from("");

    let image_topic = node
        .declare_parameter("image_topic")
        .default(empty_str_default.clone())
        .mandatory()
        .unwrap()
        .get();

    let image_path = node
        .declare_parameter("image_path")
        .default(empty_str_default.clone())
        .mandatory()
        .unwrap()
        .get();
    // no need for str sanity check: already done by rcls

    let publisher =
        node.create_publisher::<ImageMsg>(&image_topic[..], rclrs::QOS_PROFILE_DEFAULT)?;

    let mut frame_id: u32 = 1;

    while context.ok() {
        let img_buf = image::open(&image_path[..])?.grayscale();
        let nsec = node.get_clock().now().nsec;

        let header = HeaderMsg {
            stamp: builtin_interfaces::msg::Time {
                sec: (nsec / 1_000_000_000) as i32,
                nanosec: nsec as u32,
            },
            frame_id: frame_id.to_string(),
        };

        // check: https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html
        let height = img_buf.height() as u32;
        let width = img_buf.width() as u32;
        let encoding = "mono8".to_string();

        let step = height * 4;
        let data = img_buf.as_bytes().to_vec();

        let mut is_bigendian: u8 = 1;

        if cfg!(target_endian = "little") {
            is_bigendian = 0;
        }

        let message = ImageMsg {
            header,
            height,
            width,
            encoding,
            is_bigendian,
            step,
            data,
        };

        publisher.publish(&message)?;
        frame_id += 1;
        //std::thread::sleep(std::time::Duration::from_millis(500));
    }
    Ok(())
}
