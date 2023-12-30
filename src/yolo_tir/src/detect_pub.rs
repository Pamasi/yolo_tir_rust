
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


struct YoloTir{    
    class_label: Arc<[Arc<str>]>,
    conf_score : f32,
    nms_threshold: f32,
    conf_idx: usize,
    cls_idx_offset: usize,
    node: Arc<rclrs::Node>,
    img_msg: Arc< Mutex< Option<ImageMsg> > >,
    subscriber:Arc< rclrs::Subscription<ImageMsg> > ,
    publisher: Arc< rclrs::Publisher<Detection2DArray> >,
    pub model_path: Arc<str>
}


#[derive(Clone)]
struct InferenceInfo {
    pub in_width: u32,
    pub in_height: u32,
    pub n_proposal: usize,
    pub input0_shape: Vec<usize>,
}
impl YoloTir {
    pub fn new(name: &str, context: &rclrs::Context) -> Result<Self,  Box<dyn std::error::Error>> {


        let node = rclrs::Node::new(context, name)?;

        // declare parameter
        let empty_str: &str = "/";

        // Convert the string slice to Arc<str>
        let empty_str_default: Arc<str> = Arc::from("/invalid");


        // parameter must declared, use_undeclared_parameter().set(..) is wrong
        // because it overrides the launch file value (see https://github.com/ros2-rust/ros2_rust/blob/main/rclrs/src/parameter.rs)
        // once defined argument are not modified for consistency issue
        let image_topic = node.declare_parameter("image_topic")
                                .default(empty_str_default.clone())
                                .mandatory()
                                .unwrap().get();

        let detection_topic = node.declare_parameter("detection_topic")
                                .default(empty_str_default.clone())
                                .mandatory()
                                .unwrap().get();


        let model_path = node.declare_parameter("model_path")
                                    .default(empty_str_default.clone())
                                    .mandatory()
                                    .unwrap().get();

        let nms_threshold  = node.declare_parameter("nms_threshold")
                                    .default(0.50)
                                    .mandatory()
                                    .unwrap().get() as f32;

        let conf_score = node.declare_parameter("conf_score")
                                .default(0.5)
                                .mandatory()
                                .unwrap().get() as f32;

        let conf_idx = node.declare_parameter("conf_idx")
                                .default(4)
                                .mandatory()
                                .unwrap().get() as usize;
                     
        let cls_idx_offset= node.declare_parameter("cls_idx_offset")
                                .default(5)
                                .mandatory()
                                .unwrap().get() as usize;

        let classes = node.declare_parameter("classes")
                            .default_string_array(["person","bike","car","other vehicle"])
                            .mandatory()
                            .unwrap().get();                  
        // assert just for the debug
        //assert_eq!(&image_topic[..], "/tir/image");
        //assert_eq!(&detection_topic[..], "/tir/detection");

        // sanity check
        assert!(conf_score>0.0, "Confidence score must greater than 0");
        assert!(nms_threshold>0.0, "Confidence score must greater than 0");
        assert!(conf_idx>0, "Confidence idx cannot be negative");
        assert!(cls_idx_offset>0, "Confidence idx offset cannot be negative");
        assert!(classes.len()>0, "Number of classes MUST be greater than 0");

        // check that it is a onnx file
        assert_eq!(&model_path[(model_path.len()-5)..model_path.len()], ".onnx");

        let publisher =  node.create_publisher::<Detection2DArray>(&detection_topic[..], rclrs::QOS_PROFILE_DEFAULT)?;


        // clone to handle data-transfer between subscriber and publisher
        let img_msg = Arc::new(Mutex::new(None));  
        let data_sub= Arc::clone(&img_msg);

        let subscriber= node.create_subscription::<ImageMsg, _>(
                &image_topic[..],
                rclrs::QOS_PROFILE_DEFAULT,
                move |msg: ImageMsg| {
                    *data_sub.lock().unwrap() = Some(msg); 
                }
            )?;

  
        Ok(Self{
            class_label:classes,
            conf_score,
            nms_threshold,
            conf_idx,
            cls_idx_offset,
            node,
            img_msg,
            subscriber,
            publisher,
            model_path
        })
    }

    pub fn get_data(&self) -> Option<ImageMsg>{
        self.img_msg.lock().unwrap().clone()

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

        let mut offset = 0;
      
  
        let parser_endian = match img_msg.is_bigendian {
            1 => u8::from_le,
            _ => u8::from_be
        };

        //let  row_step = img_msg.step;
        //assert_eq!(img_msg.width, row_step, "step is {}", row_step);
        //println!("data len{}", img_msg.data.len());

        // normalize img
        for i in 0..img_msg.height{
            for j in 0..img_msg.width{
                let idx = (offset+j) as usize;
                let pixel_norm = (parser_endian(img_msg.data[idx]) as f32) / 255.0;
                img_data[IxDyn(&[0,0,j as usize ,i as usize ])] = pixel_norm;
            }
            offset=i*img_msg.height;
    
        }

        let input_tensor_values = vec![img_data];

        input_tensor_values


    }

}


fn main() -> Result<(), Box<dyn std::error::Error>> {



    let context = rclrs::Context::new(std::env::args())?;
    let yolo_infer = Arc::new(YoloTir::new("yolo_tir_node", &context)?);
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
            .with_model_from_file(&clone_infer.model_path[..]).unwrap();
    
        
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
            // uncomment for easy-debug
            // use std::time::Duration;
            // std::thread::sleep(Duration::from_millis(1000));

            // get data
            let msg = clone_infer.get_data();
            if msg.is_none(){
                println!("No data received!")
            }
            else{
                let img_msg = msg.unwrap().clone();
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
        
        }

    });
    rclrs::spin(yolo_infer.node.clone())?;

    Ok(())
}
