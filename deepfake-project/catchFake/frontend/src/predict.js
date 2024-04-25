// Filename - App.js
 
// Importing modules
import React, { useState, useEffect } from "react";
import "./App.css";
 
function Predict() {
    // usestate for setting a javascript
    // object for storing and using data
    const [data, setdata] = useState({
        class_id: 0,
        class_name: "",
    });
 
    // Using useEffect for single rendering
    useEffect(() => {
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
    }, []);
 
    return (
        <div className="Predict">
            <header className="Predict-header">
                <h1>Predict</h1>
                {/* Calling a data from setdata for showing */}
                <p>{data.class_id}</p>
                <p>{data.class_name}</p>

                
            </header>
        </div>
    );
}
 
export default App;