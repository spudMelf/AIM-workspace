// Filename - App.js

// Importing modules
import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
    // usestate for setting a javascript
    // object for storing and using data
    const [data, setdata] = useState({
        class_id: 0,
        class_name: "",
    });

    const [file, setFile] = useState(null);

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


    return (
        <div className="App">
            <header className="App-header">
                <h1>React and flask</h1>
                {/* Calling a data from setdata for showing */}
                

                {/* File upload input */}
                <input type="file" onChange={handleFileChange} />
                <button onClick={handleUpload}>Upload</button>
                <p>{data.class_id}</p>
                <p>{data.class_name}</p>
            </header>
        </div>
    );
}

export default App;
