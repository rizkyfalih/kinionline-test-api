# Model API
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/0993f954d899498e8c69bc63a5802132)](https://app.codacy.com/app/rizkyfalih/kinionline-test-api?utm_source=github.com&utm_medium=referral&utm_content=rizkyfalih/kinionline-test-api&utm_campaign=badger)
[![codebeat badge](https://codebeat.co/badges/72067d01-a73e-444c-9acf-84d694d41b82)](https://codebeat.co/projects/github-com-rizkyfalih/kinionline-test-api-master)
## Command
To run the server
```
Open the teminal > go to this folder directory > on terminal write 'gunicorn model-api:api'
```
To user the API
```
open postman > select POST > write 'http://127.0.0.1:8000/(ranfost/decision) > klick body > raw(json) > input json data > klik send'
```
Template for input
```
{
	"text" : "sudah disepakati"
}
```
