//
//  StartViewController.swift
//  Vision+ML Example
//
//  Created by Marco Marchesi on 1/23/19.
//  Copyright © 2019 Apple. All rights reserved.
//

import Foundation
import UIKit
import Photos

class StartViewController:UIViewController{
    
    @IBOutlet weak var startButton: UIButton!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        startButton.backgroundColor = .clear
        startButton.layer.cornerRadius = 5
        startButton.layer.borderWidth = 1
        startButton.layer.borderColor = UIColor.black.cgColor
        
        let photos = PHPhotoLibrary.authorizationStatus()
        if photos == .notDetermined {
            PHPhotoLibrary.requestAuthorization({status in
                if status == .authorized{
                    print("Authorized")
                } else {}
            })
        }
    }
}
