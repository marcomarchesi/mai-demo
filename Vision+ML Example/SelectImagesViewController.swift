//
//  SelectImagesViewController.swift
//  Vision+ML Example
//
//  Created by Marco Marchesi on 1/24/19.
//  Copyright © 2019 Apple. All rights reserved.
//

import Foundation
import UIKit


class SelectImagesViewController:UIViewController{
    
    @IBOutlet weak var selectionSlider: UISlider!
    @IBOutlet weak var selectionLabel: UILabel!
    @IBOutlet weak var selectionButton:UIButton!
    
    @IBOutlet weak var startDate:UIDatePicker!
    @IBOutlet weak var stopDate:UIDatePicker!
    
    @IBAction func sliderValueChanged(_ sender:Any){
        selectionLabel.text = "\(Float(selectionSlider.value / 100))"
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
//        selectionSlider.maximumValue = 100
        
        selectionButton.backgroundColor = .clear
        selectionButton.layer.cornerRadius = 5
        selectionButton.layer.borderWidth = 1
        selectionButton.layer.borderColor = UIColor.black.cgColor
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?)
    {
        if segue.destination is ImageClassificationViewController
        {
            let vc = segue.destination as? ImageClassificationViewController
//            vc?.limit = Int(selectionSlider.value)
            vc?.tW = Double(Float(selectionSlider.value / 100))
            vc?.aW = Double(Float(1 - selectionSlider.value / 100))
            vc?.startDate = startDate.date as NSDate
            vc?.stopDate = stopDate.date as NSDate
        }
    }
}
