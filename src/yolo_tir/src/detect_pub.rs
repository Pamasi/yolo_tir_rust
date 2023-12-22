
use anyhow::{Error,Result};

use std::env;
use std::sync::{Arc, Mutex};
use box_info::BoxInfo;
use ndarray::{s, IxDyn, ArrayView, ViewRepr};
use ndarray_stats::QuantileExt;

use onnxruntime::{environment::Environment, GraphOptimizationLevel,  session::Session, LoggingLevel,
    tensor::ort_owned_tensor::OrtOwnedTensor};

use sensor_msgs::msg::Image as ImageMsg;
use vision_msgs::msg::{BoundingBox2D, Detection2D, Detection2DArray, 
    ObjectHypothesisWithPose, ObjectHypothesis, 
    Point2D, Pose2D };


struct YoloTir<'a>{    
    class_label: [&'a str; 4],
    conf_score : f32,
    nms_threshold: f32,
    conf_idx: usize,
    cls_idx_offset: usize,
    node: Arc<rclrs::Node>,
    img_msg: Arc< Mutex< Option<ImageMsg> > >,
    subscriber:Arc< rclrs::Subscription<ImageMsg> > ,
    publisher: Arc< rclrs::Publisher<Detection2DArray> >,
}


#[derive(Clone)]
struct InferenceInfo {
    pub in_width: u32,
    pub in_height: u32,
    pub n_proposal: usize,
    pub input0_shape: Vec<usize>,
}
impl<'a> YoloTir<'a> {
    pub fn new(name: &str, context: &rclrs::Context, sub_topic: &str, pub_topic: &str,
                conf_score: f32, nms_threshold: f32,
                conf_idx: usize, cls_idx_offset: usize) -> Result<Self,  Box<dyn std::error::Error>> {


        let node = rclrs::Node::new(context, name)?;

        let publisher =  node.create_publisher::<Detection2DArray>(pub_topic, rclrs::QOS_PROFILE_DEFAULT)?;



        // clone to handle data-transfer between subscriber and publisher
        let img_msg = Arc::new(Mutex::new(None));  // (3)
        let data_sub= Arc::clone(&img_msg);

        let subscriber= node.create_subscription::<ImageMsg, _>(
                sub_topic,
                rclrs::QOS_PROFILE_DEFAULT,
                move |msg: ImageMsg| {
                    *data_sub.lock().unwrap() = Some(msg); 
                }
            )?;

 
        Ok(Self{
            class_label: ["person","bike","car","other vehicle"],
            conf_score,
            nms_threshold,
            conf_idx,
            cls_idx_offset,
            node,
            img_msg,
            subscriber,
            publisher
        })
    }

    pub fn get_data(&self) -> ImageMsg{
        self.img_msg.lock().unwrap().clone().unwrap()

    }
    pub fn publish(&self, output_vec:Vec<OrtOwnedTensor<f32, IxDyn>>, ratio_w:f32, ratio_h:f32, infer_info: Arc<InferenceInfo> )  -> Result<(), rclrs::RclrsError> {

        let mut box_list = Vec::<BoxInfo>::new();


        // prune proposal according to confidence score
        for n in 0..infer_info.n_proposal {

            let obj_score: f32 = output_vec[0][[0,n,self.conf_idx]];

            if obj_score > self.conf_score {
                // softmax
                let exp_array = output_vec[0].slice(s![0, n, self.cls_idx_offset..]).mapv(f32::exp);
                // println!("{}", exp_array);
                let prob_array = &exp_array/ exp_array.sum();

                // throw not used 'cuz data integrity check already done
                let best_idx:usize=  prob_array.argmax().unwrap();
                let best_label_score = *prob_array.max().unwrap();
                // println!("{}", prob_array);

                // better to directly access than reshaping
                let cy: f32 = output_vec[0][[0, n, 0]]*ratio_h;
                let cx: f32 = output_vec[0][[0, n, 1]]*ratio_w;
                let h: f32 =  output_vec[0][[0, n, 2]]*ratio_h;
                let w: f32 =  output_vec[0][[0, n, 3]]*ratio_w;

                let x_min: f32 = cx - 0.5 * w;
                let y_min: f32 = cy - 0.5 * h;
                let x_max: f32 = cx + 0.5 * w;
                let y_max: f32 = cy + 0.5 * h;

                box_list.push(BoxInfo::new(x_min, y_min, x_max, y_max, best_label_score, best_idx));
            }
        };

        // nms suppression
        let bboxes =  BoxInfo::nms(box_list, self.nms_threshold);
        let detections = self.pack_bbox(bboxes);

        let array_msg = Detection2DArray{
            header: Default::default(),
            detections
        };

        self.publisher.publish(&array_msg)?;

        Ok(())
        

    }
    fn pack_bbox(&self, bboxes: Vec<BoxInfo>) -> Vec<Detection2D> {
        let mut detections = Vec::<Detection2D>::with_capacity(bboxes.len());

        for bbox in bboxes{
            let hyp_ps = ObjectHypothesisWithPose{
                hypothesis: ObjectHypothesis{
                    class_id: self.class_label[bbox.label].to_string(),
                    score: bbox.score as f64,
                },
                // not used here
                pose: Default::default()
            };

            let dx = bbox.width()  as f64;
            let dy = bbox.height() as f64;
            let position = Point2D{
                x: bbox.x() as f64,
                y: bbox.y() as f64
            };

            let bbox2d = BoundingBox2D{
                center: Pose2D{
                    theta: f64::atan(dy / dx),
                    position
                },
                size_x: dx,
                size_y: dy
            };

            let item = Detection2D{
                header: Default::default(),
                results: vec![hyp_ps],
                bbox: bbox2d,
                id: "".to_string(),
            };
            
            detections.push(item);
        }

        detections
    }
    pub fn process_image(img_msg: ImageMsg, infer_info: Arc<InferenceInfo>) -> Vec<ndarray::Array::<f32,IxDyn>>{
        // easiest way to create an array
        // size is [batch, channel, width, height]
        let mut img_data = ndarray::Array::<f32,IxDyn>::zeros( IxDyn(infer_info.input0_shape.as_ref()));

        //assert!(sensor_msgs::image_encodings::isMono(img_msg.encoding));
        assert!(img_msg.encoding=="mono8");



        // normalize img
        
        for i in 0..img_msg.width{
            let offset = i*(img_msg.width-1);
            for j in 0..img_msg.height{
                let idx = (offset+j) as usize;
                let pixel_norm = (img_msg.data[idx] as f32) / 255.0;
                img_data[IxDyn(&[0,0,i as usize ,j as usize ])] = pixel_norm;
            }
    
        }

        let input_tensor_values = vec![img_data];

        input_tensor_values


    }

}


fn main() -> Result<(), Box<dyn std::error::Error>> {



    let context = rclrs::Context::new(std::env::args())?;
    let yolo_infer = Arc::new(YoloTir::new("yolo_tir_node", &context,"/tir/image", "/tir/detection",
                                        0.25, 0.45, 
                                        4, 5)?);
    let clone_infer= Arc::clone(&yolo_infer);


    // spawn a thread to publish data
    std::thread::spawn(move || -> Result<(), Box<dyn std::error::Error + Send>> {

        let environment = Environment::builder()
        .with_name("inference")
        .with_log_level(LoggingLevel::Verbose)
        .build().unwrap();
    
        let mut session =  environment
            .new_session_builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic).unwrap()
            .with_number_threads(1).unwrap()
            .with_model_from_file("param/best_seq_learn.onnx").unwrap();
    
        
        let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();
        let in_width = input0_shape[2] as u32 ;
        let in_height = input0_shape[3] as u32;
        // print!("input model size=[{},{}]",in_width, in_height);

        let output0_shape: Vec<usize> = session.outputs[0].dimensions().map(|d| d.unwrap()).collect();
        let n_proposal = output0_shape[1];

        // update info for inference

        let infer_info = Arc::new(InferenceInfo{
            in_width,
            in_height,
            input0_shape,
            n_proposal
        });

        loop {
            use std::time::Duration;
            std::thread::sleep(Duration::from_millis(1000));
            // get data
            let img_msg = clone_infer.get_data().clone();
            let input_tensor_values = YoloTir::process_image(img_msg.clone(), infer_info.clone());
            // perform the inference
            let tmp_vec= session.run(input_tensor_values);
            if  tmp_vec.is_ok() {
                let output_vec = tmp_vec.unwrap();
                let ratio_w = (img_msg.width/infer_info.in_width) as f32;
                let ratio_h= (img_msg.height/infer_info.in_height) as f32;

                clone_infer.publish(output_vec, ratio_h, ratio_w, infer_info.clone());

            }
            else{
                println!("WARNING: No valid data received");
            }
    
        }

    });
    rclrs::spin(yolo_infer.node.clone())?;

    Ok(())
}
