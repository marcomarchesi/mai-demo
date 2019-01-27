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
    
    @IBAction func sliderValueChanged(_ sender:Any){
        selectionLabel.text = "\(Int(selectionSlider.value))"
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        selectionSlider.maximumValue = 400
        
        selectionButton.backgroundColor = .clear
        selectionButton.layer.cornerRadius = 5
        selectionButton.layer.borderWidth = 1
        selectionButton.layer.borderColor = UIColor.white.cgColor
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
