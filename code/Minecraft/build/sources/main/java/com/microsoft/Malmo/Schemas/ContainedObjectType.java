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

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ContainedObjectType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ContainedObjectType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="type" use="required" type="{http://ProjectMalmo.microsoft.com}BlockOrItem" />
 *       &lt;attribute name="variant" type="{http://ProjectMalmo.microsoft.com}Variation" />
 *       &lt;attribute name="colour" type="{http://ProjectMalmo.microsoft.com}Colour" />
 *       &lt;attribute name="quantity" type="{http://www.w3.org/2001/XMLSchema}int" default="1" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ContainedObjectType")
public class ContainedObjectType {

    @XmlAttribute(name = "type", required = true)
    protected String type;
    @XmlAttribute(name = "variant")
    protected Variation variant;
    @XmlAttribute(name = "colour")
    protected Colour colour;
    @XmlAttribute(name = "quantity")
    protected Integer quantity;

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setType(String value) {
        this.type = value;
    }

    /**
     * Gets the value of the variant property.
     * 
     * @return
     *     possible object is
     *     {@link Variation }
     *     
     */
    public Variation getVariant() {
        return variant;
    }

    /**
     * Sets the value of the variant property.
     * 
     * @param value
     *     allowed object is
     *     {@link Variation }
     *     
     */
    public void setVariant(Variation value) {
        this.variant = value;
    }

    /**
     * Gets the value of the colour property.
     * 
     * @return
     *     possible object is
     *     {@link Colour }
     *     
     */
    public Colour getColour() {
        return colour;
    }

    /**
     * Sets the value of the colour property.
     * 
     * @param value
     *     allowed object is
     *     {@link Colour }
     *     
     */
    public void setColour(Colour value) {
        this.colour = value;
    }

    /**
     * Gets the value of the quantity property.
     * 
     * @return
     *     possible object is
     *     {@link Integer }
     *     
     */
    public int getQuantity() {
        if (quantity == null) {
            return  1;
        } else {
            return quantity;
        }
    }

    /**
     * Sets the value of the quantity property.
     * 
     * @param value
     *     allowed object is
     *     {@link Integer }
     *     
     */
    public void setQuantity(Integer value) {
        this.quantity = value;
    }

}
