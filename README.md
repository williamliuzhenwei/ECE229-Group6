## Website User Guide
Check Preprocessed_data1 for valid book titles<br>
To use the Find a Book function, userName = dev, password = dev1<br>
Make sure to input valid titles, otherwise will redirect to error page<br>
Please check /ECE229PPT.pdf for more infomation in case the EC2 Instance is shut off<br>

## Amazon EC2 Instance Operation Guide (For Developer)
-**Connect to the EC2 Ubuntu: ssh -i key_path.pem ubuntu@ip_address** <br>
-**Stop all python scripts: sudo pkill python** <br>
-**Start deployment flask app to 8080: python3 app.py** <br>
-**Access the webapp: http://ip_address:8080/ or ece229.ddns.net**

## Known Frontend Errors
1.Auth for the search part(Fixed)<br>
2.UI issue when switching tabs(Fixed)<br>
3.Click on Readmore leads stuck on one papge<br>


