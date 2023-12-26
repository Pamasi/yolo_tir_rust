use std::cmp::Ordering;

#[derive(Copy, Clone, Debug)]
pub struct BoxInfo{
    x1:f32,
    y1:f32,
    x2:f32,
    y2:f32,
    pub score:f32,
    pub label:usize
}

impl Eq  for BoxInfo{}

impl Ord for BoxInfo{
    fn cmp(&self, other: &Self) -> Ordering {

        if self.score < other.score {
            Ordering::Less
        }

        else if self.score > other.score {
            Ordering::Greater
        }

        else{
            Ordering::Equal
        }
    }
}


impl PartialOrd<Self> for BoxInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl PartialEq<Self> for BoxInfo{
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl BoxInfo{
    pub fn new(x1:f32, y1:f32, x2:f32, y2:f32, score:f32, label:usize) -> Self {
        Self { x1, y1, x2, y2, score, label}
    }

    pub fn width(&self) -> f32{
        self.y2- self.y1
    }
    pub fn height(&self) -> f32{
        self.x2- self.x1
    }

    pub fn x(&self) -> f32{
        self.x2
    }

    pub fn y(&self) -> f32{
        self.y2
    }


    pub fn nms(mut input_boxes:Vec<Self>, nms_threshold:f32) -> Vec<Self>{
        input_boxes.sort_by(|a, b| b.cmp(a));
        let v_area: Vec<f32> = input_boxes
            .iter()
            .map(|box_info| (box_info.x2 - box_info.x1 ) * (box_info.y2 - box_info.y1 ))
            .collect();

        let mut is_suppressed = vec![false; input_boxes.len()];

        for i in 0..input_boxes.len() {
            if is_suppressed[i] == false {
                for j in (i + 1)..input_boxes.len() {
                    if is_suppressed[j] == false {
                        let xx1 = f32::max(input_boxes[i].x1, input_boxes[j].x1);
                        let yy1 = f32::max(input_boxes[i].y1, input_boxes[j].y1);
                        let xx2 = f32::min(input_boxes[i].x2, input_boxes[j].x2);
                        let yy2 = f32::min(input_boxes[i].y2, input_boxes[j].y2);

                        let w = f32::max(0.0, xx2 - xx1 );
                        let h = f32::max(0.0, yy2 - yy1 );
                        let inter_area = w * h;
                        let iou = inter_area / (v_area[i] + v_area[j] - inter_area);

                        // remove all predictions with  the desired IoU
                        if iou >= nms_threshold {
                            is_suppressed[j] = true;
                        }
                    }
                }
            }
        }

        let out_boxes:Vec<BoxInfo> = input_boxes.iter()
            .enumerate()
            .filter(|t| !is_suppressed[t.0])
            .map( |t| *t.1).collect();
        out_boxes
    }
}


// unit-test
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_width(){
        let bbox = BoxInfo::new(1.0, 2.0, 3.0, 5.0, 0.40, 0);

        assert_eq!(bbox.width(), 3.0);
    }

    #[test]
    fn test_height(){
        let bbox = BoxInfo::new(1.0, 2.0, 3.0, 5.0, 0.40, 0);

        assert_eq!(bbox.height(), 2.0);
    }

    #[test]
    fn nms_distant_boxes(){
        let mut bbox_vec = vec![ BoxInfo::new(0.0,   0.0, 5.0, 10.0, 0.40, 0),
                                BoxInfo::new(15.0,  0.0, 30.0, 20.0, 0.20, 0),
                                BoxInfo::new(100.0,  150.0, 120.0, 200.0, 0.40, 0),
                                BoxInfo::new(180.0,  150.0, 200.0, 200.0, 0.40, 0)
        ];

        let filtered_bbox = BoxInfo::nms(bbox_vec, 0.2);

        assert_eq!(filtered_bbox, bbox_vec);
    }

    fn nms_overlapping_boxes(){
        let mut bbox_vec = vec![ BoxInfo::new(0.0,   0.0, 50.0, 50.0, 0.40, 0),
                                BoxInfo::new(10.0,  1.0, 50.0, 50.0, 0.20, 1),
                                BoxInfo::new(0.0,  0.0, 100.0, 200.0, 0.50, 0)
        ];

        let filtered_bbox = BoxInfo::nms(bbox_vec, 0.2);
        let correct_bbox = vec![BoxInfo::new(0.0,  0.0, 100.0, 200.0, 0.50, 0)];
        assert_eq!(filtered_bbox, correct_bbox);
    }
}