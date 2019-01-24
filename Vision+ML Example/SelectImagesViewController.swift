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
    
    @IBAction func sliderValueChanged(_ sender:Any){
        selectionLabel.text = "\(Int(selectionSlider.value))"
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?)
    {
        if segue.destination is ImageClassificationViewController
        {
            let vc = segue.destination as? ImageClassificationViewController
            vc?.limit = Int(selectionSlider.value)
        }
    }
}
