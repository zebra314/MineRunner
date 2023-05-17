//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
<<<<<<< HEAD
// Generated on: 2023.05.14 at 10:58:43 PM CST 
=======
// Generated on: 2023.05.17 at 07:25:31 PM CST 
>>>>>>> a3f57c8c89145179392796f00b12b77ad9360aef
//


package com.microsoft.Malmo.Schemas;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlList;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for anonymous complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType>
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element ref="{http://ProjectMalmo.microsoft.com}Time" minOccurs="0"/>
 *         &lt;element ref="{http://ProjectMalmo.microsoft.com}Weather" minOccurs="0"/>
 *         &lt;element name="AllowSpawning" type="{http://www.w3.org/2001/XMLSchema}boolean" minOccurs="0"/>
 *         &lt;element name="AllowedMobs" minOccurs="0">
 *           &lt;simpleType>
 *             &lt;list itemType="{http://ProjectMalmo.microsoft.com}EntityTypes" />
 *           &lt;/simpleType>
 *         &lt;/element>
 *       &lt;/all>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {

})
@XmlRootElement(name = "ServerInitialConditions")
public class ServerInitialConditions {

    @XmlElement(name = "Time")
    protected Time time;
    @XmlElement(name = "Weather", defaultValue = "normal")
    protected String weather;
    @XmlElement(name = "AllowSpawning", defaultValue = "false")
    protected Boolean allowSpawning;
    @XmlList
    @XmlElement(name = "AllowedMobs")
    protected List<EntityTypes> allowedMobs;

    /**
     * Gets the value of the time property.
     * 
     * @return
     *     possible object is
     *     {@link Time }
     *     
     */
    public Time getTime() {
        return time;
    }

    /**
     * Sets the value of the time property.
     * 
     * @param value
     *     allowed object is
     *     {@link Time }
     *     
     */
    public void setTime(Time value) {
        this.time = value;
    }

    /**
     * Gets the value of the weather property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getWeather() {
        return weather;
    }

    /**
     * Sets the value of the weather property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setWeather(String value) {
        this.weather = value;
    }

    /**
     * Gets the value of the allowSpawning property.
     * 
     * @return
     *     possible object is
     *     {@link Boolean }
     *     
     */
    public Boolean isAllowSpawning() {
        return allowSpawning;
    }

    /**
     * Sets the value of the allowSpawning property.
     * 
     * @param value
     *     allowed object is
     *     {@link Boolean }
     *     
     */
    public void setAllowSpawning(Boolean value) {
        this.allowSpawning = value;
    }

    /**
     * Gets the value of the allowedMobs property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the allowedMobs property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getAllowedMobs().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link EntityTypes }
     * 
     * 
     */
    public List<EntityTypes> getAllowedMobs() {
        if (allowedMobs == null) {
            allowedMobs = new ArrayList<EntityTypes>();
        }
        return this.allowedMobs;
    }

}
