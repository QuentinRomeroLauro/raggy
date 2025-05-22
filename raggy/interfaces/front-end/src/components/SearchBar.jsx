import React, { useState } from 'react';
import './SearchBar.css';

const SearchBar = ({ 
    placeholder="Search...", 
    options=["Option 1", "Option 2"],
    searchTerm,
    setSearchTerm,
    selectedOption,
    setSelectedOption,
    handleSearch,
}) => {

    const handleSearchChange = (e) => {
        setSearchTerm(e.target.value);
    };

    const handleOptionChange = (e) => {
        setSelectedOption(e.target.value);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    }


    return (
        <div className="search-bar">
            <input 
                type="text" 
                className="search-input" 
                value={searchTerm}
                onChange={handleSearchChange}
                placeholder={placeholder}
                onKeyDown={handleKeyPress}
            />
            <select className="search-dropdown" value={selectedOption} onChange={handleOptionChange}>
                {options.map((option, index) => (
                    <option key={index} value={option}>{option}</option>
                ))}
            </select>
        </div>
    );
};

export default SearchBar;
