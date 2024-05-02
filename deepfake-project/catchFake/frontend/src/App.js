// Filename - App.js

// Importing modules
import React, { useState, useEffect } from "react";
import "./App.css";
import BarLoader from "react-spinners/BarLoader";
import video from "./assets/AdobeStock_320627115.mp4"

function App() {
    // usestate for setting a javascript
    // object for storing and using data
    const [data, setdata] = useState({
        class_id: 0,
        class_name: "",
    });

    
    const [file, setFile] = useState(null);

    const [loading, setLoading] = useState(false);

    useEffect(() => {
      setLoading(true)
      setTimeout(() => {
        setLoading(false)
      }, 4200)
    }, [])
    
    // Using useEffect for single rendering
    /* useEffect(() => {
        // Using fetch to fetch the api from 
        // flask server it will be redirected to proxy
        fetch("/data").then((res) =>
            res.json().then((data) => {
                // Setting a data from api
                setdata({
                    class_id: data.class_id,
                    class_name: data.class_name,
                });
            })
        );
    }, []); */

    // Function to handle file selection
    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        setFile(selectedFile);

        if (event.target.files.length > 0) {
          setIsUploaded(true);
        } else {
          setIsUploaded(false);
        }
    };

    // Function to handle file upload
    // Function to handle file upload
    const handleUpload = () => {
      const formData = new FormData();
      formData.append("file", file);

      fetch("/predict", {
          method: "POST",
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          // Update the data state with the response
          setdata({
              class_id: data.class_id,
              class_name: data.class_name,
          });
      })
      .catch(error => {
          // Handle error
      });
    };

    const [isUploaded, setIsUploaded] = useState(false);

      



    return (
        <div className="App">
                { 
                    loading ? 

                    <BarLoader
                    height={10}
                    width={1500}
                    color={"#7e7e7e"}
                    loading={loading}
                    size={50}
                    speedMultiplier={.5}
                    aria-label="Loading Spinner"
                    data-testid="loader"
                    />

                    :

            <div>
                <header className="App-header">
                {/* Calling a data from setdata for showing */}

                {/* LOADING SCREEN */}

                    <div className="header-logo">"CATCH FAKE"</div>
            
                </header>
                <body className="App-body">
                    <video id="vidFile" src={video} autoPlay loop muted/>
                    <div className="App-Body-Content">
                        <label for="chooseFile" className={isUploaded ? "chooseFile-label-after" : "chooseFile-label-before"}>upload file</label>
                        <input id="chooseFile" type="file" onChange={handleFileChange} />

                        <button className="checkFile" onClick={handleUpload}>catch</button>
                        <p className="number">{data.class_id}</p>
                        <p className="number">{data.class_name}</p>
                    </div>
                </body>
            </div>
                }
        </div>
    );
}

export default App
